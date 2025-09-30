from ragas import SingleTurnSample 
from ragas.metrics import LLMContextPrecisionWithoutReference 

context_precision = LLMContextPrecisionWithoutReference(llm=evaluator_llm) 
sample = SingleTurnSample( 
    user_input="東京タワーの高さを教えてください。", 
    response="東京タワーの高さは333メートルです。", 
    retrieved_contexts=[ 
        "高さは333mである東京の有名なランドマークといえば、東京タワーです。" 
    ]
)

await context_precision.single_turn_ascore(sample) 