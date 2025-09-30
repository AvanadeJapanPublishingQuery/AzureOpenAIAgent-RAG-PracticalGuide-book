import asyncio
from collections import Counter
from datetime import datetime, timezone
import json
import math
import os
import textwrap
import uuid
import openai
import pandas as pd
import pyarrow as pa
from typing import List, Dict, Optional, Tuple, cast
from dataclasses import dataclass, asdict
import pyarrow.parquet as pq
import networkx as nx
from graspologic.partition import hierarchical_leiden
import numpy as np
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex, SearchField, SearchFieldDataType,
    SimpleField, SearchableField,
    VectorSearch, VectorSearchProfile, HnswAlgorithmConfiguration
)


import tiktoken

# ==== Azure OpenAI クライアントの設定 ====
api_key = os.getenv("AZURE_OPENAI_API_KEY")
api_version = os.getenv("AZURE_OPENAI_API_VERSION")
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
chat_model = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

async_openai_client = openai.AsyncAzureOpenAI(
    api_key=api_key,
    api_version=api_version,
    azure_endpoint=azure_endpoint,
)

api_key = os.getenv("AZURE_SEARCH_API_KEY")
endpoint = os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT")
async_embedding_client = openai.AsyncAzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_EMBEDDING_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_EMBEDDING_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_EMBEDDING_ENDPOINT")
)
embedding_model = os.getenv("AZURE_OPENAI_EMBEDDING_MODEL_DEPLOYMENT_NAME")

@dataclass
class EntityFinal:
    id: str
    human_readable_id: str
    title: str
    type: str
    description: Optional[str]
    text_unit_ids: Optional[List[str]]
    node_frequency: Optional[int]
    node_degree: Optional[int]
    x: Optional[float]
    y: Optional[float]

@dataclass
class RelationshipFinal:
    id: str
    human_readable_id: str
    source: str
    target: str
    description: Optional[str]
    weight: Optional[float]
    combined_degree: Optional[int]
    text_unit_ids: Optional[List[str]]

@dataclass
class CommunityFinal:
    id: str
    human_readable_id: str
    community: str
    level: str
    parent: Optional[str]
    children: Optional[List[str]]
    title: str
    entity_ids: Optional[List[str]]
    relationship_ids: Optional[List[str]]
    text_unit_ids: Optional[List[str]]
    period: Optional[str]
    size: Optional[int]

@dataclass
class CommunityReportFinal:
    id: str
    human_readable_id: str
    community: str
    level: str
    parent: Optional[str]
    children: Optional[List[str]]
    title: str
    summary: str
    full_content: str
    rank: Optional[int]
    rating_explanation: Optional[str]
    findings: Optional[str]
    full_content_json: Optional[str]
    period: Optional[str]
    size: Optional[int]

@dataclass
class CovariateFinal:
    id: str
    human_readable_id: str
    covariate_type: str
    type: str
    description: Optional[str]
    subject_id: str
    object_id: str
    status: Optional[str]
    start_date: Optional[str]
    end_date: Optional[str]
    source_text: Optional[str]
    text_unit_id: Optional[str]

@dataclass
class TextUnitFinal:
    id: str
    human_readable_id: str
    text: str
    n_tokens: int
    document_ids: Optional[List[str]]
    entity_ids: Optional[List[str]]
    relationship_ids: Optional[List[str]]
    covariate_ids: Optional[List[str]]

@dataclass
class DocumentFinal:
    id: str
    human_readable_id: str
    title: str
    text: str
    text_unit_ids: Optional[List[str]]
    creation_date: Optional[str]
    metadata: Optional[Dict[str, str]]

# --- 1. データモデルの定義 ---

@dataclass
class Document:
    id: str
    title: str
    text: str
    attributes: Optional[Dict[str, str]] = None

@dataclass
class TextUnit:
    id: str
    document_id: str
    text: str
    entity_ids: Optional[List[str]] = None
    relationship_ids: Optional[List[str]] = None
    covariate_ids: Optional[Dict[str, List[str]]] = None
    n_tokens: Optional[int] = None
    attributes: Optional[Dict[str, str]] = None
    embedding: Optional[List[float]] = None

@dataclass
class Entity:
    id: str
    title: str
    type: Optional[str] = None
    description: Optional[str] = None
    description_embedding: Optional[List[float]] = None
    name_embedding: Optional[List[float]] = None
    community_ids: Optional[List[str]] = None
    text_unit_ids: Optional[List[str]] = None
    rank: Optional[int] = None
    attributes: Optional[Dict[str, str]] = None

@dataclass
class Relationship:
    id: str
    source: str
    target: str
    description: Optional[str] = None
    description_embedding: Optional[List[float]] = None
    text_unit_ids: Optional[List[str]] = None
    rank: Optional[int] = None
    weight: float = 1.0
    attributes: Optional[Dict[str, str]] = None

@dataclass
class Covariate:
    id: str
    subject_id: str
    subject_type: str = "entity"
    covariate_type: str = "claim"
    text_unit_ids: Optional[List[str]] = None
    attributes: Optional[Dict[str, str]] = None

@dataclass
class Community:
    id: str
    title: str
    level: str
    parent: str
    children: List[str]
    entity_ids: Optional[List[str]] = None
    relationship_ids: Optional[List[str]] = None
    covariate_ids: Optional[Dict[str, List[str]]] = None
    attributes: Optional[Dict[str, str]] = None
    size: Optional[int] = None
    period: Optional[str] = None

@dataclass
class CommunityReport:
    id: str
    title: str
    community_id: str
    summary: str
    full_content: str
    rank: float = 1.0
    full_content_embedding: Optional[List[float]] = None
    attributes: Optional[Dict[str, str]] = None
    size: Optional[int] = None
    period: Optional[str] = None

# --- 2. Dataclass を Pandas DataFrame に変換する関数 ---

def dataclass_to_dataframe(objects: List) -> pd.DataFrame:
    return pd.DataFrame([asdict(obj) for obj in objects])

def append_to_parquet(df: pd.DataFrame, file_path: str):
    try:
        table = pa.Table.from_pandas(df)
        if os.path.exists(file_path):
            existing_table = pq.read_table(file_path)
            combined_table = pa.concat_tables([existing_table, table])
            pq.write_table(combined_table, file_path)
        else:
            pq.write_table(table, file_path)
        print(f"Data successfully appended to {file_path}")
    except Exception as e:
        print(f"Error appending data to {file_path}: {e}")

# --- 3. エンコーディング関数 ---

def get_encoding_fn(encoding_name: str):
    """Get the encoding model."""
    enc = tiktoken.get_encoding(encoding_name)
    
    def encode(text: str) -> list[int]:
        return enc.encode(text if isinstance(text, str) else str(text))
    
    return encode


# ドキュメントの作成
def create_documents() -> pd.DataFrame:
    documents = [
        Document(
            id="doc1",
            title="AIが医療を変える：診断の精度向上と医師の負担軽減",
            text="""
            東京タワー（とうきょうタワー、英: Tokyo Tower）は、東京都港区芝公園にある総合電波塔[3]。正式名称は日本電波塔（にっぽんでんぱとう）[4]。
1958年（昭和33年）12月23日に竣工。東京のシンボル、観光名所として知られている。

2018年度グッドデザイン賞受賞[5]。

創設者は前田久吉で、日本の「塔博士」とも称される内藤多仲らが設計（詳細は設計を参照）。
高さは333メートル[1]と広報されており、海抜では351メートル。塔脚の中心を基準とした塔脚の間隔は88.0メートル。総工費約30億円、1年半（197万4,015時間/543日間）と延べ21万9,335人の人員を要して完成した[6]。高さ270mの部分に、工事を常に最上部で仕切った黒崎建設の黒崎三朗会長[7]ら中心的な96人の名前が銅板で飾られている[8]。地上125メートル（海抜約150メートル）と223.55メートル（海抜約250メートル）に展望台を有するトラス構造の電波塔である[9]。

昼間障害標識として、頂点より黄赤色（インターナショナルオレンジ）と白色を交互に配した塗装となっている。テレビおよびFMラジオのアンテナとして放送電波を送出（#送信周波数・出力を参照）、また東日本旅客鉄道（JR東日本）の防護無線用アンテナとして緊急信号を発信するほか、東京都環境局の各種測定器なども設置されている。

完成当初は日本一高い建造物だったが、高さが日本一だったのは1968年6月26日に小笠原諸島が日本に返還され南鳥島ロランタワーと硫黄島ロランタワーに抜かれるまでの約9年半と、対馬（長崎県）のオメガタワーが解体されてから東京スカイツリーに抜かれるまでの約11年間である。自立式鉄塔に限れば、東京スカイツリーに抜かれるまでの約51年半は日本一の高さだった。2020年現在は、東京スカイツリーに次ぐ日本で2番目に高い建造物である。なお、重量については約4,000トンとされる。

株式会社 TOKYO TOWER（英: TOKYO TOWER Co.,Ltd）は、東京都港区芝公園に本社を置く東京タワーの建主であり、管理および運営を行っている。

1957年5月、「大阪の新聞王」と呼ばれ、当時、産業経済新聞社、大阪放送（ラジオ大阪）各社の社長を務め、後に関西テレビ放送[注釈 2]の社長にも就く前田久吉が日本電波塔株式会社（にっぽんでんぱとう、英: NIPPON TELEVISION CITY CORPORATION）を資本金5億円にて設立。久吉はタワーの完成とほぼ同時の1958年、産経新聞社を国策パルプ工業（現・日本製紙）社長の水野成夫に譲渡してその経営から手を引いたが、日本電波塔（東京タワー）とラジオ大阪の経営には引き続き携わった。この結果、日本電波塔は当時の産経新聞グループはもちろん、その後のフジサンケイグループからも完全に切り離されて前田家主導の同族企業となった。その名残で産経新聞グループから離脱する直前の1957年10月、文化放送やニッポン放送などと共に発足した、中央ラジオ・テレビ健康保険組合に[注釈 3]基幹会社の一社として2019年現在でも加入している[10][11]。また、ラジオ大阪も2005年にフジサンケイグループ入りするまで、前田家主導で独自の経営をしていた。

久吉は千葉県富津市鹿野山に1962年、マザー牧場を開設している関係で、2009年にはマザー牧場や日本電波塔の関連会社が同県木更津市のコミュニティFMであるかずさエフエムの株式を取得し運営しているほか[12][13]、同市や君津市を中心とする地域で整備され2010年1月に経営破綻したかずさアカデミアパークの再建スポンサーを同年8月から日本電波塔・マザー牧場・ホテルオークラ・グリーンコアの各社が務め、経営再建を行っている[14]。

1964年には敷地内に東京タワー放送センター（現・東京タワーメディアセンター）を建設し、同年開局した日本科学技術振興財団テレビ事業本部（東京12チャンネル）に賃貸した。この建物は、事業を承継したテレビ東京が1985年まで本社として使用していた。テレビ東京天王洲スタジオ完成後の2000年から日本電波塔による自主運営となり、2005年7月には子会社「東京タワー芝公園スタジオ」（のちに東京タワースタジオ）に移管され、2012年に閉鎖されるまでテレビスタジオとして利用された。なお、東京タワースタジオ閉鎖後、内部改装を施し「東京タワーメディアセンター」に名称を変更、2019年1月時点でも営業を続けている。

また、1960年代に東京タワーへのアクセスとして、日本電波塔自ら浜松町 - 東京タワー間1.2キロにモノレールの敷設を計画したが、これは実現しなかった[15]。

東京タワーは、全国FM放送協議会（JFN）のキー局エフエム東京（TOKYO FM）の大株主（学校法人東海大学に次ぐ第2位）でもある。この他にもJFNの系列局であるKiss-FM KOBEの経営破綻による新会社・兵庫エフエム放送（Kiss FM KOBE）にTOKYO FMとともに19.2%を出資[16]。また、同じくJFN加盟局のエフエム大阪（FM大阪）に20%[注釈 4]、JFN特別加盟局のInterFM897に9.4%[注釈 5]をそれぞれ出資している[17]。

2018年12月23日、東京タワーの開業60周年を機に、運営会社の商号を株式会社東京タワーに変更[18]。翌2019年10月1日には株式会社TOKYO TOWERに変更した[19]。

東京タワーの建設前、放送事業者各社局は個々に、高さ153 - 177メートルの電波塔を建設、自局の塔から放送を行っていた。当時開局していた日本放送協会（NHK）は千代田区紀尾井町に、日本テレビ放送網（NTV）は千代田区麹町に、ラジオ東京テレビ（KRT、現在のTBSテレビ）は港区赤坂に、自前の鉄塔を建設していた（詳細は電波塔集約の項を参照）。これらの高さだと、放送電波は半径70キロ程度しか届かず、100キロ[注釈 11]離れた関東平野東端の銚子や関東平野北東部[注釈 12]の水戸では満足に電波を受信することができなかった。また、受信アンテナには指向性があるため、チャンネルを変えるごとにアンテナの向きを各電波塔の方向に変えなければいけないという不便が生じた。

さらに、鉄塔の乱立は都市景観においても好ましい状況ではなく[注釈 13]、既に千代田区南西部から港区北東部にかけての一帯には先述したNHK、NTV、KRTの鉄塔が乱立する光景が広がっており、今後放送局が増加した場合、東京が電波塔だらけになる恐れがあった[57]。当時郵政省の電波管理局長であった浜田成徳をはじめとする関係者の中で、電波塔を一本化する総合電波塔を求める機運が高まっているところ、放送事業の将来性に着目した前田久吉と鹿内信隆[注釈 14]の各々によって計画され、まもなく両者の計画は一元化された。

ほかの計画案もあったが、高さ300メートルを超える案は東京タワーのみで、次に高いものは200メートル級であり、放送事業者の既存の限られた土地を利用するため、展望台のないスリムなものであった。浜田は、パリのエッフェル塔を超える世界最大の塔を造り、そこに展望台を設けて集客すれば、建設費は10年で元が取れると考えていた[要出典]。

建設地は安定した電波を供給するために巨大な電波塔の建設が可能な広さと強固な地盤を有していること、魅力ある展望台のために工場などの煙が景観を妨げないことなど厳しい条件が求められた。

当初は上野公園付近への建設も検討されたが、海抜18メートルの高台にある港区芝公園地区は基礎を打ち込むための東京礫層地盤もより浅いところにあり、国の中枢機関や各放送事業者との距離が近いなど、報道と観光の両面に恵まれた立地であった。

増上寺の境内は25区画に分割された公園に指定されており日本電波塔株式会社は「紅葉山」と呼ばれる、以前紅葉館という高級料亭のあった区画を購入した。土地の買収は増上寺の檀家総代に前田が日本工業新聞の社長時代から親交があった池貝庄太郎がおり、増上寺との間を取り持って用地買収を成功させるよう働きかけた。また、芝公園4丁目地区の周辺一帯は建物倒壊危険度、火災危険度、避難危険度を示す地域の危険度特性評価（東京都2002年実施）において「相対的に危険度の低い町」を示すAAAの評価を得ており、防災面でも電波塔の立地に適していることがのちに判明した。

タワーから西側の住民は、飛行機の衝突、交通渋滞、ゴミの増加および環境悪化で子どもに悪影響があるのではないかとタワー建築に反対の姿勢であった。東側は、当時の国鉄浜松町駅からタワーへ向かう客により潤されることを期待した。

この塔の建設に先立ち「日本電波塔株式会社」が設立された。そして、建築設計の構造力学を専門とする学者で、戦艦大和の鉄塔や名古屋テレビ塔や大阪の通天閣の設計も行い、さらに数十本におよぶラジオの電波塔を設計した実績があり、日本の塔設計の第一人者である内藤多仲、および日建設計株式会社が共同で塔の設計を行った。内藤は当時話題を提供していたドイツのシュトゥットガルトテレビ塔（英語版）（216.6メートル）を参考に鉄筋コンクリートの塔を想定した検討を行う[注釈 15]が、特に基礎に関して敷地の関係などかなりの困難が伴うとの判断から鉄塔で計算を進める方針となった[59]。

前田久吉は「建設するからには世界一高い塔でなければ意味がない。1300年も前にすでに高さ57メートルあまりもある立派な塔（五重塔）が日本人の手でできたのである。ましてや科学技術が進展した今なら必ずできる」と高さの意義を強く主張した。設計の条件として、アンテナを含めた塔の高さが380メートルで、高所に展望台を設置し、塔の下に5階建ての科学館をつくることを挙げた。東京全域に電波を送るには380メートルの高さが必要と推定されていたが、380メートルと想定して計算すると風の影響でアンテナが数メートルも大きく揺れると計算され画像が乱れる可能性が高かったため、先端のアンテナの揺れを数十センチ程度に抑え、放送に悪影響が起きず、かつ関東地方全部を守備範囲にできるぎりぎりの寸法についてさまざまに計算したところ、「全高333メートル」という数字が導き出され、偶然「3」が続く語呂合わせのような高さになった。この高さはフランス・パリのエッフェル塔の312メートル（2022年現在は330メートル[60][61]）より21メートル高く、当時の自立式鉄塔としては世界最高という条件を満たしていた。デザインはエッフェル塔に着想を得たものであるが、航空機からも見やすいよう昼間障害標識としてオレンジと白で塗装された[62]。

当初は最上部で風速90メートル、下部で風速60メートルの強風と大地震[注釈 16]に遭遇しても安全なように、軽量化に有利な電気溶接ではなく、重量がかさむが当時では確実な技術としてリベットによる接合での構造設計がなされた。風力係数は当時の建設省建築研究所の亀井勇教授に依頼し、風洞実験を行った。地震力はまだ静的解析の時代であり、鉄塔では風圧力の方が支配的であったこともあり建築基準法の地震力算定法通りk=0.16+h/400を水平震度として適用した。解析、計算はすべて手計算で、トラスの解法として一般的であったクレモナ図解法とカスティリアーノの定理が使用された。

構造計算書の最終チェックは自身の設計事務所の田中彌壽雄、日建設計の鏡才吉とともに熱海にある早稲田大学の保養所「双柿舎」に3日間缶詰状態で行われた。設計を終えた内藤は「どうかね、こんなに素晴らしい眺めはない」と言った[63]。のちに立体骨組モデル応力解析ソフトウェア“FRAN”で耐力を検証しているが、かなりの精度で一致していた[64]。また、加藤勉（東京大学名誉教授・（財）溶接研究所理事長）による「東京タワーの構造安全性について」（2007年）によって、東京タワーの構造の信頼性は高いという第三者評価を受けている[65]。当時の建築基準法では建築物の高さは最大31メートル以下と決められていたが、タワーは工作物とされ建築が可能となった。当初、地上約66メートル付近にビアレストランを設置する計画だったが、実現されることはなかった。これは、レストランにすると建築基準法に抵触したためと考えられている。

1957年5月から6月末までの約45日間でボーリング調査を行った時点で、設計は未完成であったが、短期間に完成させなければならないため、6月29日に増上寺の墓地を一部取り壊してすでに設計の決まっていた基礎部の工事が開始された。7月15日に最終的な設計図が完成し、9月21日には鉄骨の組み立てが始まった。

施工はゼネコンの竹中工務店。塔体加工は新三菱重工（現・三菱重工業）、松尾橋梁。鉄塔建築は宮地建設工業（現・宮地エンジニアリング）が請け負った。現場指揮官は直前にNHK松山放送局電波塔を担当していた同社の竹山正明（当時31歳）[66]。現場でのヘルメットの色は白：監督管理関係、黄：竹中工務店の社員、緑：鉄塔建方関係、灰：設備関係で色分けされた。

基礎は海抜0メートルの砂利層まで掘り下げて、コンクリートを打ち込み直径が2メートル、長さ15メートルで底の直径が3.5メートルのコンクリート製の円柱を1脚に8本打った（のちのデジタルアンテナ増設時に2本ずつ増設）。塔脚にはカールソン型応力計を埋めておき、脚を支えるための支柱をオイルジャッキで持ち上げて脚の傾きを調節する際など、各塔脚に加わる重量の計測に利用している。タワーは脚を広げた形をしているため、重みで脚が外へ広がろうとする力が加わるが、直径5センチの鋼棒20本を各脚に地中で対角線上に結んで防いでいる[67]。

高さ40メートルのアーチライン最上部までは長さ63メートルのガイデリックを使用し、次に80メートルまで組み立てるため、ガイデリックを地上53メートルのマンモス梁までせり上げ鉄骨を組み立てていった。80メートルからの組み立てはエレクターを用いて、鉄骨はエレベーターシャフト内を持ち上げていった[68]。

塔脚4本が地上40メートルでアーチ形に組まれたのは1957年12月だが、アーチ中央部が加工の段階で設計より15ミリ沈んでおり、梁の結合ができずに1週間原因究明に時間を費やした。しかしこの問題は、鉄骨に開けられていたリベットを差し込む穴を15ミリずらすことで解決した。

高所までの移動は、80メートルの足場まで4分で昇る2メートル四方のゴンドラ3台で対処した。

高さ141.1メートル（H.14）地点まではリベットで組み立て、それ以上は防さびのため部材に亜鉛メッキを行ったことで、ボルト接合になっている。ボルトは締めたあとに溶接して固めるが夏場の鉄塔は40 - 50度まで上昇し、とび職達を苦しめた。リベットは16万8,000本、本締めボルトは亜鉛メッキ部材の現地接合に4万5,000本使用している[69]。

アンテナの設置は当初、名古屋テレビ塔や東京スカイツリーのように塔体内を吊り上げる予定であったが、アンテナ工事は台風の多い9月に開始されたため、工事が遅れてしまいアンテナを上げる前にエレベータ設置工事を始めないと、工期に間に合わなくなってしまった。そのために、アンテナは塔本体の上に30メートルの仮設鉄塔を組んで仮設鉄塔の一面を開けておき、8つに分かれたアンテナ部材（最大14トン）を、下の部材から順に塔の外側から吊り上げていった。塔体内では吊り上げた部材に順次ボルト接合を施して組み立てていき、1958年10月14日15時47分、アンテナが塔頂部に据え付けられた。

現場鳶職人は初期に20人。仕事が増えるにつれ常時60人、タワー上部では6 - 7人で組み立てを行っていた。若頭は当時25歳の桐生五郎[注釈 17]。桐生はタワー完成翌日に、建設中に見合いをした女性と結婚式を挙げた[66]。建設には渡り職人も参加している。当時の鳶の平均日給は500円、タワーでは750円だった。高さを増すごとに強風に苦しめられたが、納期があるために風速15m/sまでは作業を実施した。建設開始時は命綱どころかヘルメットすら装着せずに高所での作業を行っており[70]、1958年6月30日10時[66]には昇っていた鳶職1人が強風に煽られて高さ61メートルから転落死し、麓にある増上寺で葬儀を行っている。

着工から1年3か月後（543日間）の1958年12月23日、延人員21万9,335人にて完成し、鉄塔本体の最上部に建設に携わった96人の技術屋たちの名前が刻まれた金属製の銘板[注釈 18]が据えられた[71]。総工費は当時の金額で30億円であった。

合計約4,000トン[69]の鋼材（鉄塔本体：SS41（旧JIS。現JISではSS400。降伏強度は240N（ニュートン）/mm2[72]）、アンテナ支持台：SHT52相当品[69] 降伏強度 325N/mm2[73]）が使用されたが、その中でも特別展望台から上の部分に使用されている鉄材の原料には、朝鮮戦争後スクラップにされたアメリカ軍の戦車約90両分が使われている[注釈 19][74]。これは当時の日本では良質の鋼材に恵まれず、またアメリカにとっても修理するより旧式戦車を売却して新型戦車を製造した方がメリットが大きかったためである[74][75]。これがバラエティ番組『トリビアの泉 〜素晴らしきムダ知識〜』で取り上げられた際に「東京タワーの特別展望台の上はさらに特別なものだった」と形容されている[74]。米軍戦車(M4, M47)を実際に解体した業者(解体スクラップは東京製鉄で形鋼になった)が問屋から聞いた話として「1000トンから1500トンが100メートルから上の東京タワーの細いアングルとして使われた」が伝わっている[76]

タワーの高さは標高18.000メートルを地盤面（G.L・Ground Line）としているため、東京湾中等潮位（T.P・TOKYO Peil）からの値を使用している[9]。

タワーは丘陵地に建っており、正面は1階が出入り口で駐車場出入り口は2階に位置している。そのためどの部分を基準としているのかがわかりにくく、タワーを訪れても基準となる目印は特に見当たらない。

一方、タワーの立面図[77] を見ると地盤面はフットタウン1階の床と同一に見える。

「東京タワー」の名称は完成直前に開かれた名称審査会で決定した。事前に名称を公募し、最終的には8万6,269通の応募が寄せられた。

一番多かった名称は「昭和塔」（1,832通）で、続いて「日本塔」「平和塔」だった。ほかには当時アメリカとソ連が人工衛星の打ち上げ競争をしていたことから「宇宙塔」、皇太子明仁親王（明仁上皇）の成婚（1959年）が近いということで「プリンス塔」という応募名称もあった。

しかし名称審査会に参加した徳川夢声が、「平凡こそ最高なり！」[78]、「ピタリと表しているのは『東京タワー』を置いて他にありませんな」と推挙し、その結果1958年10月9日に「東京タワー」に決定した。

「東京タワー」での応募は223通（全体の0.26パーセント、13位[79]）であり、抽選で神奈川県の小学校5年生女子児童に賞金10万円が贈られた。

東京タワーの完成に先行して開局していたNHK総合テレビジョン・日本テレビ放送網（NTV、以下「日本テレビ」）・ラジオ東京（1960年に東京放送（TBS）に改称。以下「TBSテレビ」）は、それぞれ自局の敷地に電波鉄塔を建設してテレビ放送を行っていた。

そのため、当初は1959年に新たに開局したNHK教育テレビジョン（1月開局。当時1ch）・日本教育テレビ（NET、現・テレビ朝日。2月開局）・フジテレビジョン（CX、3月開局。以下「フジテレビ」）が利用していた[注釈 20]（NHKは同年4月に総合テレビと教育テレビのチャンネルを交換）。翌1960年に赤坂のTBSテレビ（1月）と紀尾井町からの送信に変わっていたNHK教育テレビ（5月）も合流した。

当初はNHKと民放6局のアンテナを東京タワーに一本化するはずだったが、調整の段階で日本テレビが「採算が合わない」「アンテナの配分が不満だ」という理由で不参加になった。しかし、実際はテレビ業界の覇権を競う産経新聞創設者の前田久吉と、読売新聞中興の祖であり、日本テレビ創設者でもある正力松太郎との対立といわれる。

東京タワーの完成後も日本テレビは麹町の自社敷地内のアンテナから電波を発信し続けていたが、他局に比べて放送エリアが劣るのは否めなかった。そこで同局は自社の所有地である新宿（現在の東京メトロ副都心線・都営地下鉄大江戸線東新宿駅付近。新宿六丁目の新宿イーストサイドスクエア敷地）に東京タワーの2倍もの高さを持つ電波塔「正力タワー」の建設を計画して1968年5月10日に発表。ボーリング調査や東京都に建築申請書を提出している。

さらにNHKも「NHKが民放に恒久施設を借りた例がない。そんなことをしては聴視者に責任がもてない」と1969年7月2日、渋谷のNHK放送センター敷地内に高さ600メートルの電波塔を建てる計画を発表する。これを受けて放送事業の監督官庁であった郵政省は、利害関係者である日本電波塔（東京タワーの会社）を交えて話し合うことを指示するが、日本テレビは「あくまでも実行する」と発表[80]。

しかし正力没後の翌1970年に、メインアンテナを麹町の本社から東京タワーへ移転することになった[注釈 21]。その際、TBSが自社の予備スペースを日本テレビに譲ることで、メインアンテナのスペースを確保している[注釈 22]。これにより全放送局が東京タワーに集約された。

建設時の塗装は地上からH.14（地上141.1m）までは鋼材を工場にてサンドブラストしてから下塗りまで塗装し、現地搬入時に2回目の下塗りをした。接合部のリベット頭・部材エッジ部は予め下塗り塗装による増し塗りが行われ中塗り、上塗りにはフタル酸樹脂系塗料を用いて6行程行われた。H.14からH.27（地上252.65m）までは工場にて鋼材を酸洗いしてから溶融亜鉛メッキを施し現地搬入、建方、本締めボルト接合後にジンククロメート系さび止めペイントを塗装した。中塗り、上塗りは鉄鋼材と同様である[69]。また、建築当時はインターナショナルオレンジがさび止めの鉛丹と間違えられ、完成した後も最終的にはどのような色になるのかまだペインティングの途中だと思っている人がいた。

東京湾からの潮風による腐食の防止とインターナショナルオレンジの白化現象が目立ち美観を保つ上でほぼ5年に1度の周期で約1年かけて外観塗装を補修しており、1回目（1965年）は磯部塗装が担当し2回目（1970年）以降は平岩塗装が一貫して請け負っている。平岩塗装の平岩高夫会長は前田久吉から会うたびに「鉄はサビさせてはあかんからな。大事に守ってくれ」と言われている[81]。

鉄塔本体の塗装工区は大展望台を境に2つに分け上部は秋に、下部は翌年の春に施工している。作業時間は日の出から営業を開始する9時までに限定されていたが現在は7時を過ぎると通勤者が増えるため、2時から7時までの5時間で作業を行う。

最上部のアンテナを除くH.27の塔体の上から順に木製の丸太で足場を組み、まずケレン落としと呼ばれる下作業をして下塗り、中塗り、上塗りと3工程が行われるがこれは全てハケを使い人の手によって塗られる。上部のアンテナは電波を送信するため足場を組むわけにはいかず、放送電波が停止した時間に職人がアンテナをよじ上って作業をした。総塗装面積94,000m2に使うペンキの量は34,000リットルとなり、延べ約4200人が作業に当たる。使用する塗料は一斗缶に置き換えて縦に積み重ねると、東京タワーの2倍の高さになる[82]。金属ではなく木製の丸太を使用するのは近隣への防音対策と電波への影響を避けるためであり、丸太の総数は1万本以上となる。以前にグラスファイバーのパイプを使用しようとしたことがあったがジョイント部分のクランプの使い勝手が悪く、木製の丸太に落ち着いている。また、塗装期間中のライトアップは右写真のようにタワーに影ができる。

塗装工事の最終日には、タワーのマークと職人一人ひとりの名前が刻まれた五百円玉大の記念メダルが職人に手渡された。

全面塗替は第1回が1965年、第2回が1970年、第3回が1976年、第4回が1980年、第5回が1986年、第6回が1991年、第7回が1996年、第8回が2001年、第9回が2007年に行われているがその都度塗り足しをしているため、9回の作業で完成時より約1mm塗装が厚くなっている[83]。そのあと、2013年ごろに第10回目の塗り替えが行われ、2018年3月から2019年夏ごろまでの予定で、第11回目の塗り替えが行われている。塗料の性能向上により、今後は塗り替えが7年周期に延長される予定である[84]。このような定期的な塗装メンテナンスにより鉄骨が錆から保護されている限り、タワー本体は半永久的に存続可能であるとされている[85]。

航空法によりストロボのような白色航空障害灯を常時点滅させれば現在のペイントを変更することは可能だが周囲には住宅や高速道路があり、住民への迷惑、運転者への安全を考えるとペインティングの変更は考えにくい。2008年現在は7等分の塗り分けだが、建設当時から1986年までは11等分に塗り分けていた。また、大展望台の外壁は現在は白色だが、1996年まではインターナショナルオレンジだった。

なお、塗装作業の一部にビートたけしと北野大の父である北野菊次郎が携わっていた。

この塔の売り上げは観光による収入が5割を超えている。東京近辺を目的地とする修学旅行などにおける定番の行き先として定着している。開業翌年の1959年に当時上野動物園が持つ年間入場者記録360万人を大幅に抜き、来塔者513万人（1 - 12月）を記録[65]。一時落ち込みはあったが、現在は年間300万人が来塔している。

タワーは地方や海外からの観光客が多く利用し、地元の東京都民、特に港区民は「東京タワーは『おのぼりさん』が行くところ」と登ったことがないという人もいる。そのため日本電波塔株式会社では港区の小学生の招待や、社員の意識改革を行い若手デザイナーを起用し、イルミネーションなどを企画して来塔者数を増やした[86]。

2008年12月23日の開業50周年に併せ、以下の事業を行った。

コンセプトは「ゆったり楽しむ壮大な東京の景色」である。2018年6月の時点で段階的にリフォーム中。

なお、「大展望台」は2018年3月3日に改修とともに「メインデッキ」に改称された。

特別展望台（現：トップデッキ）のトイレは洋式で男性用、女性用が用意されている。公衆電話はあるが売店、自動販売機はない。ゴミ入れはあるが穴が缶くらいの大きさのため、基本的にゴミは持ち帰りである。かつては大展望台2階からは「風と光のプロローグ」というテーマで装飾された階段とエスカレーターを使ってトップデッキ行きエレベーターホールまで行ったが、現在は違う。このため車いすは利用できない。AEDは設置されている。2018年3月の大規模修繕時に「特別展望台」は「トップデッキ」に改称された。

リフォーム後はジオメトリックミラーやLED照明で美しく仕上がっている。内装デザインはKAZ SHIRANE、内装デザイン補助プログラムは合同会社髙木秀太事務所が担当した。

また、チケットは事前にインターネットで予約するか、1階のチケット売り場で予約する形態になっている。観光する際はツアーをするような形態で、「トップデッキツアー」と呼んでいる。

タワーの下にある5階建ての観光・娯楽施設で、以前は「タワービル」と呼び科学館でもあった。タワーのおもりとして設計されている。過去にはTEPCOタワーランドやフジテレビタワープラザといった東京電力やフジテレビのショールームも入居していた。

2023年時点の施設[88]。2022年4月に国内最大規模のe-sportパーク「RED° TOKYO TOWER」がオープンした[89][90]。3階から5階のRED° TOKYO TOWERエリアは入場チケットが必要（施設内アトラクションは無料）。

イルミネーションは1958年12月21日に実験的に灯され、開業から20日間毎晩点灯された。その後は日曜、祝日の前夜に点灯し1964年の東京オリンピック中は連夜点灯。これが好評であったために1965年のクリスマスイブから連夜の点灯となった。電球は鉄塔の四隅に5m間隔で250灯配置していたが随時増えてゆき、1976年には696灯（塔体384灯、アンテナ88灯、特別展望台96灯、大展望台128灯）となった。電球が切れた場合はその都度交換しに行くのは手間がかかるため、ある程度切れたら交換していたが所々光が途切れている部分は目立ち、ある芸能人から「地方から東京に帰ってきてタワーを見ると、電球が切れていて気になる」と指摘を受けている[94]。

電気料金は2008年12月時点では投光器の精度がアップしたことと中間部照明器具に消費電力の少ないLEDを使用していることにより、約1万8520円と約25%の省電力となっている[97]。

過去に「乳がん撲滅キャンペーン（ピンクリボンデー）」で桃色、映画『マトリックス・リローデッド』のプロモーションやアイルランドと日本の外交関係50周年の記念として緑色、地上デジタル放送のプロモーションや世界糖尿病デーで青色のライトアップを実施したこともある。

2000年から年末年始（クリスマス終了から1月中旬まで）に大展望台の窓ガラスの外側に西暦の数字を装飾している。時間は16時30分 - 翌朝8時まで。ただし、大晦日は西暦表示の切り替え作業のため、0時から8時までの点灯となる。

2005年12月の地上デジタル放送のプロモーションで「地デジ」の文字や、2016年の東京オリンピック招致でオリンピックの色を使った「Tokyo」「2016」の文字を装飾していた時期があった。クリスマスには大展望台にピンクのハートが点灯される。

2007年の第58回NHK紅白歌合戦では総合優勝を決めるにあたり従来のそれぞれの審査カテゴリー別の得票最多チームをボールで数える「玉入れ方式」を行わず、東京タワーのライトアップに拠って最終成績を決めるという試みを行った。

2008年クリスマスの期間中（2010年は12月1日 - 26日）、毎日19時半に一旦ランドマークライトアップが消灯し大展望台にピンクの ハートマーク や大展望台から地上へ繋がるフラッシュライトがクリスマスソングに合わせて光のショー「東京タワークリスマス・ライトダウンストーリー」を展開する。

スポーツ大会で日本を応援する時や、優勝した場合には特別ライトアップが行われる時があり、夏・冬のオリンピック・パラリンピックで日本人選手・チームが金メダルを獲得した時[98]、2009年のワールド・ベースボール・クラシックで侍JAPANが連覇を決めた時、サッカー日本代表がFIFAワールドカップ南アフリカ大会出場を決定した時、更には2011年にFIFA女子ワールドカップドイツ大会でサッカー日本女子代表が優勝した時には祝賀のライトアップを行った。

2011年に、石井の提案により東日本大震災で落ち込んだ日本を応援するべく、大展望台に8400個のLEDで「GANBARO NIPPON」の文字装飾を4月11日から16日まで行った[99]。電力は太陽光発電でまかなわれ、その後ハートマークの装飾も実施した。また、6月20日には世界難民デーにあわせ、国際連合に使用されている色にライトアップされた。

2013年7月18日から7月21日までの4日間は、同月19日から10月6日まで開催の「藤子・F・不二雄展」に合わせ、青・赤・黄を基調としたドラえもんカラー（青は体色、赤は首輪、黄は鈴）にライトアップされた[100]。また、9月3日のドラえもんの誕生日に合わせ、同年9月2日から9月4日までの3日間にも再び同色にライトアップされた[101]。

2015年11月15日は2日前に発生したパリ同時多発テロ事件への哀悼の意味を込めて、東京スカイツリーとともにフランスの国旗と同じ青・白・赤にライトアップされた[102]。

2017年9月23日は当日に番組終了を迎えたテレビ朝日系『SmaSTATION!!』が番組開始から16年間、東京タワーをバックにオープニングの挨拶をしてくれたことに対する感謝の意味を込めて、22時30分から2時間限定で番組テーマカラーである青色にライトアップされた[103][104]。

なお、ライトアップには事前に決められた年間カレンダー[注釈 27]に沿って、デザイナーなど専門家の監修やプログラミングを経て行われている。大展望台の窓文字も職人が窓にLEDのパネルをはめたり、配線などの作業が必要なことから10日前後の日時が必要となる[106]。

設備の点検・工事などの夜間作業が行われる場合を除きライトアップの照明は0時に消灯されていたが「東京タワーのライトアップが消える瞬間を一緒に見つめたカップルは永遠の幸せを手に入れる」との噂が広まり、0時前になるとライトダウンの瞬間を見ようとする多くのカップルが集まるようになった[107]。これは漫画『部長島耕作』で恋人の誕生日にケーキに立てるろうそくの数を1本少なく間違えた主人公・島耕作がタワーの灯を巨大なろうそくに見立てて0時の消灯と同時に吹き消して見せるシーンがあり、これが伝説の由来となっている可能性が高い[108]。

ライトダウンは施設管理部電気課の職員がスイッチを操作するがアンテナの設備点検や工事があるために0時以降もライトダウンしないと「なぜ今日は消えないのか」と毎回問い合わせがあるため、現在は0時[注釈 28]に消灯して再度0時半頃に点灯することになっている。スイッチは回転式で、大展望台の上と下を別々に操作することができる。ランドマークライトからダイアモンドヴェールに切り替わる際にもライトダウンを見ることができる。

また、2011年3月12日以降、東日本大震災（東北地方太平洋沖地震）のため自主的なライトダウン[109] を同年5月10日まで終日行った[110]。5月11日から13日の3日間は白色のダイヤモンドヴェール（追悼の光）を実施し、その後は自然エネルギーを取り入れるなどさらなる省エネに努めライトアップを再開している。

東京都心部にある観光地であり、地上300mを超える光の塔となるタワーには写真撮影をする地方や海外からの観光客が跡を絶たない。人とタワーを一緒に撮影する場合、敷地内ではタワーが近すぎ、また巨大であるために身をかがめて撮影する人々が見受けられる。子供を撮影するに至っては寝転がって撮影する人もいる。そのために東京タワーには正面入り口左側に記念撮影用の立て看板が設置されている。この写真は順光となる2階出入り口側の2号塔脚付近で撮影されており、団体客の集合写真も2階出入り口付近で撮影される。

正面入り口側からの日中の撮影では、逆光で顔が黒くつぶれるためにフラッシュが必要となる。ただしその場合、タワーは暗く写るので注意が必要だが最近のカメラは自動で逆光補正を行う機種もあるためHDRやオートで撮影するのも手である。

地上デジタル放送用の送信アンテナをどこに設置するかについては多摩地区、上野地区、秋葉原地区などから誘致提案が出された。しかしサービスエリアや航空路との関係などの面でいずれも決定的ではなく、2003年12月1日からの関東地区での地上デジタル放送開始に対応する仮の措置として従来アナログテレビ放送を行って来た東京タワーの施設を拡張する形で設置されることになった。このため大展望台の直下、135 - 145mの高さに送信設備室を増築し特別展望台の上の塔体最上部に直径13m、高さ11mの筒型のアンテナを設置した。

デジタルテレビ電波はアナログテレビ電波に比べて送信所に近い地域（強電界地域）では受信電波障害の範囲が狭くなるが、送信所から遠い地域（中、弱電界地域）ではアナログテレビ電波同様に障害が発生する[111]。

2013年（平成25年）まで、東京タワーでは地上デジタル放送波の送信を行っていた。これは暫定的な措置であり、2011年（平成23年）7月24日の地上デジタル放送全面移行後、今後さらに超高層建築物が建設されることも踏まえ、東京タワーの高さでは受信電波障害の問題を解決できず、（2009年の時点では）首都圏域すべて（アナログテレビの放送区域）をカバーし切れないと言われていた。

この問題は、2000年代初頭から既に想定されていたが、2003年（平成15年）12月1日の地上デジタル放送開始には間に合わなかった。しかしその頃から、東京タワーに代わる新しい電波塔の建設が検討され始め、同年12月17日に東京タワーの電波障害を解消するためにNHKと民放キー局で構成する「在京6社新タワー推進プロジェクト」が発足。候補地については東京23区および近隣の数都市が名乗りを挙げて協議を重ねた結果、2006年に新塔の建設予定地が墨田区押上にある東武鉄道の貨物操車場跡地に決定。ここに新塔「東京スカイツリー」（以前は仮に「第2東京タワー」や「新東京タワー」「すみだタワー」などと呼んでいた）を建設することとなった。2008年7月14日に着工し地上デジタル放送への全面移行後の2012年2月29日に完成、同年5月22日に開業した。高さは634mで電波塔としては世界一を更新し地上350m地点に展望デッキ（第1展望台・フロア340-350）、450m地点に展望回廊（第2展望台・フロア445-450・最高地点451.2m）を設置する[112]。建設費400億円。総事業費は約650億円[113]。本放送は2013年5月31日9時に開始された（東京タワーからの送信所切替）。

一方、東京タワーを管理する日本電波塔は、2007年（平成19年）9月21日、放送局各局のデジタルテレビ完全移行後に、塔頂部にある現在のアナログテレビ用スーパーターンスタイルアンテナを撤去し、その場所へデジタルテレビ用アンテナを設置することで、アンテナ位置を80 - 100m上方に移動する方針を主軸とした、東京スカイツリーへの対抗案（東京タワーイノベーション計画）を打ち出し各テレビ局に打診した。80m改修案の場合、タワーの高さは変わらず、費用は約40億円とされ、各局の放送設備もそのまま流用できるなど、東京スカイツリーの建設に比べると圧倒的にコストを低く抑えられた。なお、材質には軽量の炭素繊維などを用いた場合、大掛かりな補強工事の費用は必要なく、改装費用はさらに減額することが可能であった[114][65]。

2010年（平成22年）9月27日、日本電波塔はNHKおよび民放テレビキー局5社との間でテレビ送信（NHKはFMも含む）の東京スカイツリーへの移行後、災害時などで東京スカイツリーから電波が送れない場合の予備電波塔として、東京タワーの利用契約を結んだ[115]。

2018年（平成30年）9月30日、地上デジタルテレビ放送への完全統合後も、東京タワーから送信を続けてきた放送大学地上波放送が、放送衛星を利用した衛星放送に統合されたことを受けて廃止となることから、東京タワーのテレビ放送の常時送信が終了となった。なおラジオ放送のinterfm、TOKYO FMは継続して東京タワーからの送信を行っているほか、地デジ・NHK-FMの予備送信所機能も継続されている[116]。

地上デジタルテレビ放送FMラジオ放送V-Low帯マルチメディア放送

3素子2L5段15面 2系統NHK-DG・NHK-DE・EX-D・CX-D・TBS-D・NTV-D・TX-D【予備送信所】2L2段4面（北方向のみ1段）TOKYO FMSG8段4面NHK-FM【予備送信所】2L1段4面InterFM2L4段4面UD-FM

【メイン送信所】TOKYO FM（TFM）InterFM897【予備送信所】NHK東京（テレビ・FM）日本テレビ（NTV）TBSテレビフジテレビ（CX）テレビ朝日（EX）テレビ東京（TX）

デジタル：10kWFMラジオ：10kWマルチメディア放送：10kW

約14,000,000世帯

発信される電波は、関東平野一円の半径100km圏を範疇とする。

建設当初からのアナログテレビ放送送信所（送信機室）はタワービルの5階にある。ただし5階は各放送局の送信設備などが置かれた機械室となり、保安上や安全上（感電事故等防止）の面から、関係者以外は立入禁止である。

なお、1997年3月10日にフジテレビで放送された『FNNスーパータイム』でフジテレビの河田町からお台場への電波の引っ越しのニュースを放送した時や2008年9月にフジテレビONE（当時:フジテレビ739）で放送された『ばら・す』では、フジテレビのアナログテレビ送信設備の一部が放送されたほか、2011年3月にNHK総合・NHK BS2で放送された『ブラタモリ』（第2シリーズ）ではNHK教育テレビのアナログテレビ送信設備の一部が放送されている。また、2011年7月24日深夜（25日未明）にテレビ朝日で放送された『ANN NEWS&SPORTS』や同年7月25日朝にNHK総合で放送された『NHKニュースおはよう日本』でも技術職員によるアナログ放送の完全停波作業のニュースが伝えられた際、それぞれのアナログテレビ送信設備の一部が放送されている（NHKは総合テレビの設備）。さらに、2013年5月31日9時に行われた地上デジタル放送の東京スカイツリーへの送信所完全移転の際には、関東ローカルニュースにて、東京タワーからの電波を止める瞬間の模様を放送した際に各局のデジタル送信所の一部が放送された。

送信機は2層を使って設置されており上階にNHKのテレビジョン放送2波分、放送大学と地上デジタル音声放送の実用化試験放送用（後1者は廃局済）、下階に民放テレビジョン放送5局分、それぞれの送信機を設置。なお、TOKYO MXは単独設置（アナログ送信機と共用）であった（移転済）。送信機は共通仕様に基づき設計された固体化水冷式で最大出力10kWが得られる。

地上デジタル放送送信設備室はフットタウンと大展望台を結ぶ階段からでないと入ることができない。なお、前述のとおり放送大学以外の局については2013年5月31日9時以降、予備送信所となっている。

特別展望台とアナログ放送用アンテナ部分の中間、高さ260 - 280mのところに直径13m・高さ12mの円筒形のアンテナを設置した。これは「3素子型2L双ループアンテナ5段15面4系統」といわれるもので、ループ型アンテナを構成するエレメントを10段30面に配置している。そして赤に塗装された上5段分から3波、白に塗られた下5段分から4波が送信される。このアンテナは前述の通り、当初は親局として使用されていたが放送大学以外の局に関しては2013年5月31日9時以降は東京スカイツリーからの送信が不可能になったときの予備アンテナとなっている。放送大学もBSデジタル放送に一本化するため、2018年9月をもって地上波での放送を終了しており、同年10月以降は東京タワーを親局として運用している地上デジタル放送は無い。

地上デジタル音声放送用のアンテナは特別展望台の直下、高さ約245mのところにプレートパラボラアンテナが設置されている（2011年3月31日の全局廃局に伴い、同年6月にアンテナ撤去）。これらの設備追加により塔は420tも重量が増加したとの案内が行われていた。2001年、タワーへのアンテナおよび送信機室の設置に伴って構造安全性が再検討され2003年から2005年春にかけて塔の構造材に鋼板による補強[118] と塔脚一本につきコンクリート杭（アースドリル工法・直径3m、深さ約18.5mの基礎杭）が2本ずつ増設[119] された。

2003年の運用開始当初はアナログテレビの混信を避けるため出力の抑制・指向性が掛けられていたが、アナアナ変換による対策がこの地域で完了した2005年までに無指向性・所定の出力となった。

アナログ放送のアンテナは塔頂部からNHK総合と教育（STアンテナ6段にて二重給電）、テレビ朝日、フジテレビ、TBSと日本テレビ（併設）、テレビ東京と放送大学（併設。ただし、放送大学のUHF送信アンテナはスキュー配列[注釈 29]）（以上広域放送、NHK教育のみ全国放送）の順で塔頂部からH.27（地上27番目の鉄骨の水平材、桁）までのゲイン塔に設置されていた。ただし、TOKYO MXは開設が放送大学より後なのと県域放送（東京都のみでの放送）のため一段低い場所にあった。保守、管理をしていたのは電気興業。設計は同社の鈴田豊次（当時25歳）ら新米ばかりの若いチームだった。アンテナは送信波長の関係から太くできないため、直径17cmのステンレス丸棒を溶接したものを使用していた。ここから送信された電波は160km離れた栃木県那須湯本まで届いた[120]。

地上アナログテレビジョン放送の終了に伴い2011年7月25日0時に送信を終了し、地上アナログ放送用アンテナは2012年7月までに撤去された。

これらの放送局のアンテナは頂上部のTOKYO FM、特別展望台直下のNHK-FM（予備送信所）、interfmの順に設置されている。

特別展望台直下のアンテナは、NHK-FM・TOKYO FM・J-WAVEの3局が、1つのアンテナを共用していたがNHK-FMとJ-WAVEについては東京スカイツリーに送信設備を移転、TOKYO FMも後述の通り頂上部に移している。NHK-FMは引き続き予備送信所として残るが、J-WAVEは本社のある六本木ヒルズ森タワー屋上の予備送信所を継続利用するため使用されていない。以前はNHK-FMとInterFMの中間に放送大学のアンテナが設置されていたが、前述のテレビ放送同様BSデジタル放送への一本化のため2018年9月をもって地上波での放送を終了している。

2013年2月11日より、TOKYO FMのアンテナが、NHK東京のアナログテレビのあったタワー頂上部に新設された[39]。当初の計画では2012年1月に、従来のNHKアナログテレビのアンテナを転用して送信する予定だったが、前述の東日本大震災によるアンテナ破損があったため、旧アンテナを撤去して新設することとなった[121]。

InterFMでは開局時期が遅いこともあり、当初は地上150mの大展望台の直上という低い位置、ラジオNIKKEIの中継アンテナと同じ場所にあった。その後、首都圏の高層建築物の増加に伴って受信環境が悪化していたことから、アンテナを従来より高くした上で、かつ同じ周波数では混信が発生するため、同時に送信周波数76.1MHzを変更することとなった。2015年6月24日に総務省から変更許可を受け[122]、同月26日より試験放送を開始[45]、30日18時より本放送を開始した[46]。10月31日まで移行期間として新旧周波数でサイマル放送を行い、11月1日より送信周波数を89.7MHzに一本化した。

この他、以前は大展望台の直下にAMラジオのニッポン放送のラジオ中継用のアンテナがあった。

一般視聴者向けの放送アンテナ以外に、テレビ局は素材を遠方の取材先から演奏所に送るためのマイクロ波による中継システムを持つ。この塔には、送られて来たマイクロ波を受信するアンテナが、FM用送信アンテナ群の直下から大展望台にかけて設置され、遠隔操作で取材地方向に向けることができる。また、タワー自体による死角ができるので、対向する2個所1組で運用。これらのアンテナで受信した電波は、映像専用の回線を通じて各放送局の演奏所に送られる。

キー局が共同取材で素材を融通し合う場合があるが、アナログ放送時代は「タワー分岐」と呼ばれる作業により、送信所への中継回線の予備回線を利用して各局に映像素材を配信できるようになっていた。デジタル放送開始後は、ネクシオンが提供する映像伝送サービス「ネクシオンHD分岐」に移行した。

なお、光ファイバーによる大容量の伝送回線網が日本全国をカバーしたため、放送局相互用マイクロ波回線（NTT中継回線）は廃止された。

アナログVHFから移行した以下の7局は2003年12月1日から本放送を行ってきたが、2013年5月31日9時を以って東京スカイツリーに主送信所を移転したことに伴い東京スカイツリーから送信できなくなった場合の予備送信所に移行した。

2012年4月23日の放送開始から主送信所を東京スカイツリーに移したため、東京スカイツリーから送信できなくなった場合の予備送信所に移行した[136]。

廃止日：2013年5月12日（東京スカイツリーへの移転のため）、2018年10月30日（BSデジタル放送移行のため）

（授業実施予定地域[142][143]）

廃止日：2011年7月24日（放送法による一斉免許失効のため）

廃止日：2011年3月31日（実用化試験放送終了のため）[147]

廃止日：2012年4月23日（東京スカイツリーへの移転のため、J-WAVE）、2015年10月31日（下記周波数の運用終了のため、InterFM）、2018年10月30日（BSデジタル放送移行のため、放送大学）

（授業実施予定地域）[151][152]

廃止日：2020年3月31日（運用終了のため）[153]

日本国外の建造物については東京タワーの高さを超える主要建造物のみ掲載。

1963年4月15日に東京タワーを使用した社名として「東京タワー観光バス」が設立された。同社は1969年10月2日に国際興業に買収され、1972年4月1日に合併された。

東京タワーはその時代や東京という地理的背景を説明するためのシンボルとして、建設以来実にさまざまな小説や映像作品の中に登場している。本項では、数多の作品の中からその一部を紹介する。

※怪獣ものは、下方に別項「怪獣もの」としてまとめる。

※怪獣ものは、下方に別項「怪獣もの」としてまとめる。

怪獣と東京タワーは縁が深い。最初に東京を襲った大怪獣はゴジラであるが、その際には東京タワーを破壊していない[注釈 33]。しかし1958年の東京タワー完成後は数多くの怪獣映画に登場し、特にテレビで怪獣ものが流れるようになってからは頻繁に破壊されるようになった[157]。

なお、日本の怪獣映画のスターであり、最初の映画怪獣でもあるゴジラが東京タワーを倒したという印象が広く浸透している。たとえば、下記の小松左京の小説地球になった男にも「型通り東京タワーをへしおり」と書かれてある[158]。同様の例として、清原なつのの少女漫画作品である『ゴジラサンド日和』ではリバイバルのゴジラを見に行ったかつてのカップルを描写したシーンでゴジラが「うりゃっ」というかけ声とともに東京タワーを叩き折っている場面が描かれている[要ページ番号]。これらは少なくとも「怪獣は東京タワーを破壊するもの」とのイメージが実在したことを示すものである。

その他にも『三大怪獣 地球最大の決戦』（1964年）、『地球攻撃命令 ゴジラ対ガイガン』（1972年）、『ゴジラ FINAL WARS』（2004年）、『巨神兵東京に現わる』（2012年）など多くの怪獣映画で東京タワーが登場し、かつ破壊されている。

東京タワーは開業当初から観光施設としての性質を持ちミニチュア（タワー模型）やプラモデル、ペナント、絵葉書などのおみやげ品が用意されていたが近年上記の『Tokyo Tower』や『東京タワー 〜オカンとボクと、時々、オトン〜』、『ALWAYS 三丁目の夕日』などで別の意味で注目されるようになり一般の店舗での関連商品が登場した。

2011年度までの「イメージガール」に代わる新しいキャンペーンキャラクターとして2012年度より「東京タワーアンバサダー」を制定。

TBSホールディングス

1フジテレビジョン、ニッポン放送、ポニーキャニオンなどを子会社に持つ認定放送持株会社。2フジ・メディア・ホールディングスは、系列局の仙台放送を連結子会社化、基幹局の北海道文化放送、関西テレビ放送、テレビ新広島を筆頭に複数の系列局を持分法適用関連会社化している。3フジテレビジョン、ニッポン放送、ポニーキャニオン、産業経済新聞社、文化放送などを中心に構成するメディア・コングロマリット。4フジテレビジョンと国際的戦略提携を締結。

2003年 鮎河ナオミ - 2004年 杉浦美帆 - 2005年・2006年 小林さくら - 2007年 遥香 - 2008年 折井あゆみ - 2009年・2010年 梅田彩佳 - 2011年 小林香菜

2012年 吉松育美 - 2013年 Mei - 2014年・2015年 金ヶ江悦子

            """
        )
        ]
    return dataclass_to_dataframe(documents)


async def create_text_chunks_async(row: pd.Series, chunk_size: int, id_prefix: str, chunk_start_id: int, encode) -> list[TextUnit]:
    """ 1つのドキュメントを非同期でチャンク化し、リストとして返す。 """
    doc_id = row["id"]
    text_chunks = await asyncio.to_thread(textwrap.wrap, row["text"], width=chunk_size)
    
    chunked_data = []
    for i, chunk in enumerate(text_chunks):
        token_count = await asyncio.to_thread(encode, chunk)
        chunked_data.append(TextUnit(
            id=f"{id_prefix}_{chunk_start_id + i}",
            document_id=doc_id,
            text=chunk,
            n_tokens=len(token_count)
        ))
    
    return chunked_data


async def create_text_units(documents: pd.DataFrame, chunk_size: int = 100, id_prefix: str = "chunk", encoding_name: str = "o200k_base") -> pd.DataFrame:
    """ 分割されたテキストチャンクを含むデータフレームを非同期で作成する。 """
    encode = get_encoding_fn(encoding_name)
    tasks = [
        create_text_chunks_async(row, chunk_size, id_prefix, idx * 1000, encode)
        for idx, row in documents.iterrows()
    ]
    results = await asyncio.gather(*tasks)
    chunked_data = [item for sublist in results for item in sublist]
    return dataclass_to_dataframe(chunked_data)


async def create_final_documents_async(documents: pd.DataFrame, text_units: pd.DataFrame) -> pd.DataFrame:
    """All the steps to transform final documents asynchronously."""
    exploded = await asyncio.to_thread(lambda: (
        text_units.explode("document_id")
        .loc[:, ["id", "document_id", "text"]]
        .rename(columns={
            "document_id": "chunk_doc_id",
            "id": "chunk_id",
            "text": "chunk_text",
        })
    ))
    joined = await asyncio.to_thread(lambda: exploded.merge(
        documents,
        left_on="chunk_doc_id",
        right_on="id",
        how="inner",
        copy=False,
    ))
    docs_with_text_units = await asyncio.to_thread(lambda: joined.groupby("id", sort=False).agg(
        text_unit_ids=("chunk_id", list)
    ))
    rejoined = await asyncio.to_thread(lambda: docs_with_text_units.merge(
        documents,
        on="id",
        how="right",
        copy=False,
    ).reset_index(drop=True))
    rejoined["id"] = rejoined["id"].astype(str)
    rejoined["human_readable_id"] = rejoined.index + 1
    return rejoined

async def extract_entities(text: str) -> Tuple[List[Entity], List[Relationship]]:
    prompt = f"""
    以下のテキストからエンティティとリレーションシップを抽出してください。
    
    テキスト: {text}
    
    出力は以下のJSON形式で返してください：
    {{
      "entities": [
        {{"title": "エンティティ名", "type": "エンティティタイプ", "description": "エンティティの詳細説明", }}
      ],
      "relationships": [
        {{"source": "ソースエンティティ", "target": "ターゲットエンティティ", "description": "ソースエンティティとタターゲットエンティティの関係の説明", }}
      ]
    }}
    """

    try:
        response = await async_openai_client.chat.completions.create(
            model=chat_model,
            messages=[
                {"role": "system", "content": "You are an AI assistant."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
        )
        
        response_content = response.choices[0].message.content.strip()
        parsed_data = json.loads(response_content)

        # エンティティを作成（idを自動生成）
        entities = [
            Entity(id=str(uuid.uuid4()), **entity)  # IDをUUIDで自動生成
            for entity in parsed_data.get("entities", [])
        ]

        # リレーションシップも同様に処理
        relationships = [
            Relationship(id=str(uuid.uuid4()), **relationship)  # IDをUUIDで自動生成
            for relationship in parsed_data.get("relationships", [])
        ]

        return entities, relationships   
    except Exception as e:
        print(f"Error extracting entities: {e}")
        raise

async def extract_graph(text_units: pd.DataFrame, text_column: str = "text"):
    try:
        # １行ずつ逐次処理
        for _, row in text_units.iterrows():
            text = row[text_column]
            entities, relationships = await extract_entities(text)

            # １行毎に複数のentitiesが取得できる為、関係性を表現するためにIDを付与
            for entity in entities:
                print(entity)
                entity.__dict__["text_unit_ids"] = [row["id"]]
                entity_df = pd.DataFrame([entity.__dict__])
                append_to_parquet(entity_df, "entities.parquet")
            # １行毎に複数のrelationshipsが取得できる為、関係性を表現するためにIDを付与
            for relationship in relationships:
                print(relationship)
                relationship.__dict__["text_unit_ids"] = [row["id"]]
                relationship_df = pd.DataFrame([relationship.__dict__])
                append_to_parquet(relationship_df, "relationships.parquet")

    except Exception as e:
        print(f"Error extracting graph: {e}")

# Pandas DataFrame形式のrelationshipからnetworkxのグラフを作成
async def create_graph_from_pandas_relationship(relationships_df: pd.DataFrame) -> nx.Graph:
    return await asyncio.to_thread(nx.from_pandas_edgelist, relationships_df, "source", "target")

# ノードの次数を計算
async def compute_node_degrees_from_graph(graph: nx.Graph) -> pd.DataFrame:
    return await asyncio.to_thread(lambda: pd.DataFrame(
        [
            { "title": node, "degree": int(degree) } for node, degree in graph.degree
        ]
    ))

async def layout_graph(graph: nx.Graph) -> pd.DataFrame:
    position = await asyncio.to_thread(nx.spring_layout, graph)
    layout_data = [{"title": node, "x": position[node][0], "y": position[node][1]} for node in graph.nodes]
    return pd.DataFrame(layout_data)

def compute_node_frequency_from_pandas_relations(relationships_df: pd.DataFrame) -> pd.DataFrame:
    frequency_df = (relationships_df["source"].value_counts() + relationships_df["target"].value_counts()).reset_index()
    frequency_df.columns = ["title", "frequency"]
    return frequency_df.fillna(0).astype({"frequency": int})

async def finalize_entities():
    entities_df = pd.read_parquet("entities.parquet")
    relationships_df = pd.read_parquet("relationships.parquet")

    graph = await create_graph_from_pandas_relationship(relationships_df)

    layout_df, node_degree_df = await asyncio.gather(
        layout_graph(graph),
        compute_node_degrees_from_graph(graph)
    )

    node_frequency_df = compute_node_frequency_from_pandas_relations(relationships_df)

    def merge_dataframes(entities_df: pd.DataFrame, node_degree_df: pd.DataFrame, node_frequency_df: pd.DataFrame, layout_df: pd.DataFrame) -> pd.DataFrame:
        return entities_df.merge(
            node_degree_df, on="title").merge(
                node_frequency_df, on="title").merge(
                    layout_df, on="title").reset_index(drop=True)
    
    def process_missing_value(df: pd.DataFrame) -> pd.DataFrame:
        return df.fillna(0).astype({"degree": int, "frequency": int})
    
    final_entities_df = merge_dataframes(entities_df, node_degree_df, node_frequency_df, layout_df)
    final_entities_df = process_missing_value(final_entities_df)
    append_to_parquet(final_entities_df, "final_entities.parquet")
    print("Finalized entities complete.")

def compute_edge_combined_degree(
    edge_df: pd.DataFrame,
    node_degree_df: pd.DataFrame,
    node_name_column: str,
    node_degree_column: str,
    edge_source_column: str,
    edge_target_column: str,
) -> pd.Series:
    """Compute the combined degree for each edge in a graph."""

    def degree_colname(column: str) -> str:
        return f"{column}_degree"
    
    def join_to_degree(df: pd.DataFrame, column: str) -> pd.DataFrame:
        degree_column = degree_colname(column)
        result = df.merge(
            node_degree_df.rename(
                columns={node_name_column: column, node_degree_column: degree_column}
            ),
            on=column,
            how="left",
        )
        result[degree_column] = result[degree_column].fillna(0)
        return result


    output_df = join_to_degree(edge_df, edge_source_column)
    output_df = join_to_degree(output_df, edge_target_column)
    output_df["combined_degree"] = (
        output_df[degree_colname(edge_source_column)]
        + output_df[degree_colname(edge_target_column)]
    )
    return cast("pd.Series", output_df["combined_degree"])

async def finalize_relationship():
    relationships_df = pd.read_parquet("relationships.parquet")

    graph = await create_graph_from_pandas_relationship(relationships_df)
    node_degrees_df = await compute_node_degrees_from_graph(graph)

    final_relationships = relationships_df.drop_duplicates(subset=["source", "target"])

    final_relationships["combined_degree"] = compute_edge_combined_degree(
        final_relationships,
        node_degrees_df,
        node_name_column="title",
        node_degree_column="degree",
        edge_source_column="source",
        edge_target_column="target",
    )

    append_to_parquet(final_relationships, "final_relationships.parquet")
    print("Finalized relationships complete.")

def compute_leiden_communities(graph: nx.Graph) -> Tuple[Dict[int, Dict[str, int]], Dict[int, int]]:
    """
    グラフに対して Leiden 法を適用し、コミュニティ（クラスタ）を階層的に分類する関数。

    【処理の流れ】
    1. グラフの最大連結成分 (LCC: Largest Connected Component) を抽出
    2. `hierarchical_leiden` を使ってグラフを階層的にクラスタリング
    3. クラスタ情報を `results` に格納 (レベルごとのクラスタ情報)
    4. 親クラスタと子クラスタの関係を `hierarchy` に格納

    【戻り値】
    - `results` (dict[int, dict[str, int]])  
        { level番号: { ノード名: クラスタID } }
        - 各レベルごとに、ノードがどのクラスタに属するかを記録

    - `hierarchy` (dict[int, int])  
        { クラスタID: 親クラスタID }
        - クラスタの階層構造を管理（親がない場合は `-1`）

    【例】
    結果のデータ構造:
    results = {
        0: { 'Node1': 1, 'Node2': 1, 'Node3': 2, 'Node4': 2 },  # 0階層目のクラスタ情報
        1: { 'Node1': 10, 'Node2': 10, 'Node3': 11, 'Node4': 11 } # 1階層目のクラスタ情報
    }
    hierarchy = {
        1: -1,  # クラスタ1は親なし（ルートクラスタ）
        2: -1,  # クラスタ2は親なし
        10: 1,  # クラスタ10はクラスタ1の子
        11: 2   # クラスタ11はクラスタ2の子
    }
    """

    # (1) 最大連結成分 (LCC) のノードを取得
    largest_component_nodes = max(nx.connected_components(graph), key=len)

    # (2) LCC のサブグラフを作成（オリジナルのグラフをコピー）
    graph = graph.subgraph(largest_component_nodes).copy()

    # (3) Leiden法を用いた階層的クラスタリングを実行
    community_mapping = hierarchical_leiden(graph, max_cluster_size=10, resolution=1.2)

    # クラスタ情報を格納する辞書
    results: Dict[int, Dict[str, int]] = {}  # {レベル: {ノード: クラスタID}}
    hierarchy: Dict[int, int] = {}  # {クラスタID: 親クラスタID}

    # (4) 各クラスタを `results` に格納し、親クラスタとの関係を `hierarchy` に記録
    for partition in community_mapping:
        
        # レベルごとのクラスタ情報を格納
        results.setdefault(partition.level, {})[partition.node] = partition.cluster
        
        # クラスタの階層関係を格納（親クラスタが存在しない場合は `-1`）
        hierarchy[partition.cluster] = partition.parent_cluster if partition.parent_cluster is not None else -1

    return results, hierarchy


def cluster_graph(community_mapping: dict[int, dict[str, int]],hierarchy:dict[int, int]) -> List[tuple[int, int, int, list[str]]]:

    levels = sorted(community_mapping.keys())
    clusters: dict[int, dict[int, list[str]]] = {}
    for level in levels:
        clusters[level] = {}
        for node_id, raw_community_id in community_mapping[level].items():
            if raw_community_id not in clusters[level]:
                clusters[level][raw_community_id] = []
            clusters[level][raw_community_id].append(node_id)
    
    results = []
    for level, communities in clusters.items():
        for cluster_id, nodes in communities.items():
            results.append((level, cluster_id, hierarchy.get(cluster_id, -1), nodes))
    return results


async def create_community():
    entities_df = pd.read_parquet("entities.parquet")
    relationships_df = pd.read_parquet("relationships.parquet")

    try:

        # networkxのグラフを作成
        graph = await create_graph_from_pandas_relationship(relationships_df)

        # leiden法によってグラフのコミュニティを計算
        community_mapping, hierarchy = compute_leiden_communities(graph)

        # グラフコミュニティからクラスタリングデータを作成
        clustered_data = cluster_graph(community_mapping, hierarchy)

        # クラスタリングデータをデータフレームに変換
        community_df = pd.DataFrame(
            clustered_data, columns=pd.Index(["level", "community", "parent", "title"])
            ).explode("title")
        community_df["community"] = community_df["community"].astype(int)

        # コミュニティデータとエンティティデータを紐付け
        entity_ids = community_df.merge(entities_df, on="title", how="inner")
        entity_ids = (
            entity_ids.groupby("community").agg(entity_ids=("id", list)).reset_index()
            )

        # 最大のコミュニティ階層レベルを取得
        max_level = community_df["level"].max()
        all_grouped = pd.DataFrame(
            columns=["community", "level", "relationship_ids", "text_unit_ids"]
        )

        # 各レベルのコミュニティを処理
        for level in range(max_level + 1):
            # (1) 現在の `level` に属するコミュニティを取得
            communities_at_level = community_df.loc[community_df["level"] == level]

            # (2) `relationships_df` (リレーション情報) を `source` で `communities_at_level` と結合
            sources = relationships_df.merge(
                communities_at_level, left_on="source", right_on="title", how="inner"
            )

            # (3) `source` の結合結果を `target` で再度 `communities_at_level` と結合
            targets = sources.merge(
                communities_at_level, left_on="target", right_on="title", how="inner"
            )

            # (4) 同じコミュニティに属する `source` と `target` のリレーションのみを抽出
            matched = targets.loc[targets["community_x"] == targets["community_y"]]

            # (5) `text_unit_ids` を展開（1つのセルに複数値がある場合、それぞれを別の行として扱う）
            text_units = matched.explode("text_unit_ids")

            # (6) `community_x` ごとに `relationship_ids` と `text_unit_ids` を集約
            grouped = (
                text_units.groupby(["community_x", "level_x", "parent_x"])
                .agg(relationship_ids=("id", list), text_unit_ids=("text_unit_ids", list))
                .reset_index()
            )

            # (7) カラム名を `final_communities` に合わせて変更
            grouped.rename(
                columns={
                    "community_x": "community",
                    "level_x": "level",
                    "parent_x": "parent",
                },
                inplace=True,
            )

            # (8) `all_grouped` に今回のレベルのデータを追加
            all_grouped = pd.concat([
                all_grouped,
                grouped.loc[
                    :, ["community", "level", "parent", "relationship_ids", "text_unit_ids"]
                ],
            ])

        # (9) `relationship_ids` と `text_unit_ids` を `set()` で一意にして並び替え
        all_grouped["relationship_ids"] = all_grouped["relationship_ids"].apply(
            lambda x: sorted(set(x))
        )
        all_grouped["text_unit_ids"] = all_grouped["text_unit_ids"].apply(
            lambda x: sorted(set(x))
        )

        # (10) `entity_ids` を追加して `final_communities` を作成
        final_communities = all_grouped.merge(entity_ids, on="community", how="inner")

        # (11) 各コミュニティにユニークな ID を付与
        final_communities["id"] = [str(uuid.uuid4()) for _ in range(len(final_communities))]

        # (12) `human_readable_id` を `community` の値に設定
        final_communities["human_readable_id"] = final_communities["community"]

        # (13) `title` を `"Community X"` の形式に設定
        final_communities["title"] = "Community " + final_communities["community"].astype(str)

        # (14) `parent` を整数型に変換
        final_communities["parent"] = final_communities["parent"].astype(int)

        # (15) `parent` ごとに `children` をリスト化
        parent_grouped = cast(
            "pd.DataFrame",
            final_communities.groupby("parent").agg(children=("community", "unique")),
        )

        # (16) `final_communities` に `children` 情報を追加
        final_communities = final_communities.merge(
            parent_grouped,
            left_on="community",
            right_on="parent",
            how="left",
        )

        # (17) `children` が NaN の場合は空リストに変換
        final_communities["children"] = final_communities["children"].apply(
            lambda x: x if isinstance(x, np.ndarray) else []  # type: ignore
        )

        # (18) `period` に現在の日付を設定（ISO-8601形式）
        final_communities["period"] = datetime.now(timezone.utc).date().isoformat()

        # (19) `size` を `entity_ids` の数として設定
        final_communities["size"] = final_communities.loc[:, "entity_ids"].apply(len)


    except Exception as e:
        print(f"Error creating community: {e}")
    
    append_to_parquet(final_communities, "final_community.parquet")
    print("Finalized community complete.")


community_report_schema = pa.schema([
    ("id", pa.string()),
    ("human_readable_id", pa.string()),
    ("community", pa.string()),
    ("level", pa.string()),
    ("parent", pa.string()),
    ("children", pa.list_(pa.string())),
    ("title", pa.string()),
    ("summary", pa.string()),
    ("full_content", pa.string()),
    ("rank", pa.int64()),
    ("rating_explanation", pa.string()),
    ("findings", pa.string()),
    ("full_content_json", pa.string()),
    ("period", pa.string()),
    ("size", pa.int64()),
])

def append_community_report_to_parquet(record: dict, file_path: str):
    try:
        if record is None or not isinstance(record, dict) or len(record) == 0:
            print("⚠️ Skipping empty or invalid record")
            return

        # 型変換（childrenなど）
        children = record.get("children")
        if isinstance(children, np.ndarray):
            record["children"] = [str(c) for c in children.tolist()]
        elif isinstance(children, list):
            record["children"] = [str(c) for c in children]
        elif children is None:
            record["children"] = []
        else:
            record["children"] = [str(children)]

        # DataFrame & 型補正
        df = pd.DataFrame([record])
        for col in community_report_schema.names:
            if col not in df.columns:
                df[col] = None
            typ = community_report_schema.field(col).type
            if pa.types.is_string(typ):
                df[col] = df[col].astype("string")
            elif pa.types.is_int64(typ):
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
            elif pa.types.is_list(typ):
                df[col] = df[col].apply(lambda x: x if isinstance(x, list) else [])

        # 追記保存（append=True）
        if os.path.exists(file_path):
            existing_df = pd.read_parquet(file_path)
            combined_df = pd.concat([existing_df, df], ignore_index=True)
            combined_df.to_parquet(file_path, index=False)
        else:
            df.to_parquet(file_path, index=False)

        print(f"Appended 1 record to {file_path}")

    except Exception as e:
        print(f"Error writing to {file_path}: {e}")


async def create_community_report():

    try:
        entity_df = pd.read_parquet("entities.parquet")
        relationship_df = pd.read_parquet("relationships.parquet")
        communities_df = pd.read_parquet("community.parquet")

        COMMUNITY_REPORT_PROMPT = """
            以下のテキストから、コミュニティに関する簡潔なレポートを作成してください。
            レポートには以下の情報を含めてください：
            - title: コミュニティを代表する短いタイトル
            - summary: コミュニティ内の主要なエンティティと関係性の要約
            - findings: エンティティ間の関係や注目点を簡潔にまとめたリスト（3件程度）

            テキスト: {input_text}

            出力は以下の形式でJSONとして返してください：

            {{
            "title": "コミュニティタイトル",
            "summary": "コミュニティの全体概要（200文字程度）",
            "findings": [
                {{
                    "summary": "インサイト1の要約",
                    "explanation": "インサイト1の説明"
                }},
                {{
                    "summary": "インサイト2の要約",
                    "explanation": "インサイト2の説明"
                }},
                {{
                    "summary": "インサイト3の要約",
                    "explanation": "インサイト3の説明"
                }}
            ]
            }}
        """

        def build_report_record(community_row, parsed_response: dict, full_content: str) -> dict:
            return {
                "id": community_row["id"],
                "human_readable_id": community_row.get("human_readable_id"),
                "community": community_row["community"],
                "level": community_row["level"],
                "parent": community_row.get("parent"),
                "children": community_row.get("children"),
                "title": parsed_response.get("title"),
                "summary": parsed_response.get("summary"),
                "full_content": full_content,
                "rank": parsed_response.get("rating"),  # ratingがない場合はNoneになる
                "rating_explanation": parsed_response.get("rating_explanation"),
                "findings": json.dumps(parsed_response.get("findings", []), ensure_ascii=False, indent=2),
                "full_content_json": json.dumps(parsed_response, ensure_ascii=False, indent=2),
                "period": community_row.get("period"),
                "size": community_row.get("size"),
            }
        
        async def process_single_community(row):
            try:
                community_id = row["community"]
                entity_ids = row["entity_ids"]
                relationship_ids = row["relationship_ids"]

                # エンティティとリレーションの抽出
                community_entities = entity_df[entity_df["id"].isin(entity_ids)]
                community_relationships = relationship_df[relationship_df["id"].isin(relationship_ids)]

                # プロンプト構築
                entity_section = "Entities\n\nid,entity,description\n"
                entity_section += "\n".join([
                    f'{e["id"]},{e["title"]},{e["description"] or "No description"}'
                    for _, e in community_entities.iterrows()
                ])

                relationship_section = "Relationships\n\nid,source,target,description\n"
                relationship_section += "\n".join([f'{r["id"]},{r["source"]},{r["target"]},{r["description"] or "No description"}'for _, r in community_relationships.iterrows()])

                input_text = f"{entity_section}\n\n{relationship_section}"

                prompt = COMMUNITY_REPORT_PROMPT.replace("{input_text}", input_text)

                response = await async_openai_client.chat.completions.create(
                    model=chat_model,
                    messages=[
                        {"role": "system", "content": "You are an AI assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.2,
                )

                content = response.choices[0].message.content.strip()
                parsed = json.loads(content)
                record = build_report_record(row, parsed, content)
                return record
            
            except Exception as e:
                print(f"Error processing community {community_id}: {e}")
                return None

        tasks = [process_single_community(row) for _, row in communities_df.iterrows()]
        results = await asyncio.gather(*tasks)
        for record in results:
            if record is not None:
                append_community_report_to_parquet(record, "community_reports.parquet")

        # None を除いて DataFrame に
        valid_results = [r for r in results if r is not None]
        return pd.DataFrame(valid_results)

    except Exception as e:
        print(f"Error creating community report: {e}")


async def get_embedding(text: str) -> list[float]:
    res = await async_embedding_client.embeddings.create(
        model=embedding_model,
        input=[text]
    )
    return res.data[0].embedding

def is_valid_embedding(embedding):
    return (
        isinstance(embedding, list)
        and all(isinstance(x, float) and not math.isnan(x) for x in embedding)
    )

def validate_documents(documents, label=""):
    for i, doc in enumerate(documents):
        if not isinstance(doc["id"], str):
            print(f"[{label}] id is not string at index {i}: {doc['id']} ({type(doc['id'])})")
        if not isinstance(doc["content"], str):
            print(f" [{label}] content is not string at index {i}")
        if not is_valid_embedding(doc["embedding"]):
            print(f" [{label}] embedding is invalid at index {i}")
        if "community_ids" in doc:
            if not isinstance(doc["community_ids"], list) or not all(isinstance(cid, str) for cid in doc["community_ids"]):
                print(f" [{label}] community_ids invalid at index {i}: {doc.get('community_ids')}")

def build_index(name, extra_fields=[]):
    try:
        fields = [
            SimpleField(name="id", type=SearchFieldDataType.String, key=True),
            SearchableField(name="content", type=SearchFieldDataType.String),
            SearchField(
                name="embedding",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=1536,
                vector_search_profile_name="hnsw_profile"
            )
        ] + extra_fields
        index = SearchIndex(
            name=name,
            fields=fields,
            vector_search=VectorSearch(
                algorithms=[HnswAlgorithmConfiguration(name="hnsw_algorithm")],
                profiles=[VectorSearchProfile(name="hnsw_profile", algorithm_configuration_name="hnsw_algorithm")]
            )
        )

        SearchIndexClient(endpoint, AzureKeyCredential(api_key)).create_or_update_index(index)
        print(f"Index '{name}' created successfully")
    except Exception as e:
        print(f"Error creating index {name}: {e}")
        raise

async def create_community_report_index():
    try:
        # === グローバル検索用 community_report_index ===
        reports = pd.read_parquet("community_reports.parquet")

        def build_report_text(row):
            try:
                findings = json.loads(row["findings"])
                f_text = "\n".join(f["summary"] + "。" + f["explanation"] for f in findings)
            except Exception:
                f_text = ""
            return f"{row['title']}\n{row['summary']}\n{f_text}"

        reports["content"] = reports.apply(build_report_text, axis=1)

        print("Generating embeddings for community reports...")
        report_embeddings = await asyncio.gather(*[
            get_embedding(row["content"]) for _, row in reports.iterrows()
        ])
        reports["embedding"] = report_embeddings
        build_index("community-report-index")

        report_documents = []
        for _, row in reports.iterrows():
            doc = {
                "id": str(row["id"]) if pd.notna(row["id"]) else str(uuid.uuid4()),
                "content": str(row["content"]),
                "embedding": [float(x) for x in row["embedding"]] if isinstance(row["embedding"], list) else []
            }
            report_documents.append(doc)

        validate_documents(report_documents, label="report")
        SearchClient(endpoint, "community-report-index", AzureKeyCredential(api_key)).upload_documents(report_documents)
        print("Community report documents uploaded.")

    except Exception as e:
        print(f" Error creating search index: {e}")

# セマフォで並列数制限
semaphore = asyncio.Semaphore(10)

async def get_embedding_with_retry(text, retries=3, delay=2):
    for i in range(retries):
        try:
            async with semaphore:
                return await get_embedding(text)
        except Exception as e:
            print(f"[Retry {i+1}] Failed to get embedding: {e}")
            await asyncio.sleep(delay)
    print("[Error] Failed after retries, returning empty embedding.")
    return []

async def create_entity_index():
    try:
        entities = pd.read_parquet("final_entities.parquet")
        communities = pd.read_parquet("community.parquet")

        # entity_id → community_id のマッピングを作成
        entity_to_community = {}
        for _, row in communities.iterrows():
            for eid in row["entity_ids"]:
                entity_to_community.setdefault(eid, []).append(str(row["id"]))

        def build_entity_text(row):
            return f"{row['title']}\n{row['description'] or ''}"

        entities["content"] = entities.apply(build_entity_text, axis=1)
        

        print("Generating embeddings for entities...")

        entity_embeddings = await asyncio.gather(*[
            get_embedding_with_retry(row["content"]) for _, row in entities.iterrows()
        ])
        entities["embedding"] = entity_embeddings

        build_index(
            "entity-index",
            extra_fields=[
                SearchField(
                    name="community_ids",
                    type=SearchFieldDataType.Collection(SearchFieldDataType.String),
                    filterable=True
                )
            ]
        )

        entity_documents = []
        for _, row in entities.iterrows():
            eid = str(row["id"])
            doc = {
                "id": eid,
                "content": str(row["content"]),
                "embedding": [float(x) for x in row["embedding"]] if isinstance(row["embedding"], list) else [],
                "community_ids": entity_to_community.get(eid, [])
            }
            entity_documents.append(doc)

        validate_documents(entity_documents, label="entity")
        SearchClient(endpoint, "entity-index", AzureKeyCredential(api_key)).upload_documents(entity_documents)
        print(" Entity documents uploaded.")

    except Exception as e:
        print(f"[Fatal Error] create_entity_index failed: {e}")
        raise

async def graph_search(query: str, top_k: int = 5):
    search_client_local = SearchClient(endpoint, "entity-index", AzureKeyCredential(api_key))
    search_client_global = SearchClient(endpoint, "community-report-index", AzureKeyCredential(api_key))

    print(f"\nStep 1: Embedding query and searching for relevant entities...")
    query_embedding = await get_embedding(query)

    # === Step 1: エンティティのベクトル検索 ===
    entity_vector_query = {
        "vector": query_embedding,
        "k": top_k,
        "fields": "embedding",
        "kind": "vector",
        "profile": "hnsw_profile"
    }

    entity_results = search_client_local.search(
        search_text="",  # ← テキスト検索を無効化
        vector_queries=[entity_vector_query],
        select=["id", "content", "community_ids"],
        top=top_k
    )

    entity_results = list(entity_results)
    if not entity_results:
        print("No relevant entities found.")
        return

    for i, res in enumerate(entity_results, 1):
        print(f"[Entity {i}] {res['content'][:100]}...")

    # === Step 2: エンティティから community_id を集約 ===
    print(f"\nStep 2: Inferring most relevant community_id from entity results...")
    community_counter = Counter()
    for res in entity_results:
        for cid in res.get("community_ids", []):
            community_counter[cid] += 1

    if not community_counter:
        print("No community_id found in entities.")
        return

    inferred_community_id, count = community_counter.most_common(1)[0]
    print(f"Inferred community_id: {inferred_community_id} (appeared {count} times)")

    # === Step 3: エンティティ内容を結合して refined embedding を作成 ===
    combined_text = " ".join(res["content"] for res in entity_results)
    refined_embedding = await get_embedding(combined_text)

    # === Step 4: グローバルな community_report に対して検索 ===
    print(f"\nStep 3: Searching related community reports based on entity context...")

    report_vector_query = {
        "vector": refined_embedding,
        "k": top_k,
        "fields": "embedding",
        "kind": "vector",
        "profile": "hnsw_profile"
    }

    report_results = search_client_global.search(
        search_text="",
        vector_queries=[report_vector_query],
        select=["id", "content"],
        top=top_k
    )

    report_results = list(report_results)
    if not report_results:
        print("No relevant community reports found.")
        return

    for i, res in enumerate(report_results, 1):
        print(f"[Report {i}] {res['content'][:120]}...\n")

    return query, entity_results, report_results

async def generate_answer_from_graph_search(query: str, entities: list, reports: list) -> str:
    if not entities and not reports:
        return "検索に該当する情報が見つかりませんでした。"

    entity_texts = [res["content"] for res in entities]
    report_texts = [res["content"] for res in reports]

    context = "\n\n---\n\n".join(entity_texts + report_texts)

    prompt = f"""
    以下のコンテキストに基づいて、ユーザーの質問に答えてください。
    
    ユーザーの質問: {query}

    コンテキスト:
    {context}
    """


    print("\n[LLM] Generating answer from retrieved context...")
    response = await async_openai_client.chat.completions.create(
            model=chat_model,
            messages=[
                {"role": "system", "content": "You are an AI assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
    )

    return response.choices[0].message.content



async def run_pipeline():
    documents_df = create_documents()
    print("--- Documents ---")
    print(documents_df)

    text_units_df = await create_text_units(documents_df)
    print("\n--- Text Units ---")
    print(text_units_df)

    final_documents_df = await create_final_documents_async(documents_df, text_units_df)
    print("\n--- Final Documents ---")
    print(final_documents_df)

    await extract_graph(text_units_df)

    await finalize_entities()

    await finalize_relationship()

    await create_community()

    await create_community_report()

    await create_community_report_index()
    await create_entity_index()

    query, entities, reports =  await graph_search("東京タワーとかずさアカデミアパークの関係は？")
    answer = await generate_answer_from_graph_search(query, entities, reports)
    print("\n=== 回答 ===\n")
    print(answer)


# 実行
asyncio.run(run_pipeline())