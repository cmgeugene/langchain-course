from dotenv import load_dotenv

load_dotenv()

from langchain_ollama import ChatOllama
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langsmith import traceable

MAX_ITERATIONS = 10
MODEL = "qwen3:8b"

# -------- Tools (@tool decorator)-------------
@tool
def get_product_price(product:str) -> float:
    """ Look up the price of a product in the catalog. """
    print(f"    >> Executing get_product_price(product='{product}')") # LLM이 실제로 도구를 실행했는지 확인하기 위한 로그
    prices = {"laptop" : 1299.90, "headphones":149.55, "keyboard": 59.98} # 도구가 사용할 실제 데이터-> 프로덕션에선 여기에 sql쿼리, api 호출이 들어감
    return prices.get(product, 0) # 도구 사용 결과를 LLM에게 전달

@tool
def apply_discount(price:float, discount_tier:str) -> float:
    """ Apply a discount tier to a price and return the final price. Available tiers : bronze, silver, gold."""
    print(f"    >> Executing apply_discount(price={price}, discount_tier='{discount_tier}')")
    discount_percentages = {"bronze" : 5, "silver":12,"gold": 23}
    discount = discount_percentages.get(discount_tier, 0)
    return round(price * (1-discount/100), 2)

# ---- Agent Loop ----
@traceable(name="langchain agent loop") # langsmith에서 추적하기 위한 데코레이터
def run_agent(question:str):
    tools = [get_product_price, apply_discount]  # LLM 전달용 도구 리스트
    tool_dict = {t.name: t for t in tools}  # 실제 실행용 도구 딕셔너리 (like 가상함수 테이블)

    llm = init_chat_model(f"ollama:{MODEL}", temperature=0) # 챗 모델 선언
    llm_with_tools = llm.bind_tools(tools) # 모델에 도구 포함 객체 생성

    print(f"Question: {question}")
    print("=" * 60)

    messages = [
        SystemMessage(
            "You are a helpful shopping assistant."
            "You have acces to a product catalog tool. and a discount tool.\n\n"
            "STRICT RULES - you MUST follow these exactly:\n"
            "1.NEVER guess or assume any product price."
            "2.Only call apply_discount AFTER you have recieved a price from get_product_price. Pass the exact price returned by get_product_price - do NOT pass a made-up number.\n"
            "3.NEVER calculate discounts yourself using math. ALWAYS use the apply_discount tool.\n"
            "4.If ther user does not specify a discount tier, ask them which tier to use - do NOT assume one."
        ),
        HumanMessage(content=question)
    ]

    # LLM의 추론 결과가 점점 쌓이다가 최종 결과 반환하는 구조
    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n------ Iteration {iteration} --------")
        ai_message = llm_with_tools.invoke(messages)
        tool_called = ai_message.tool_calls

        # 도구 호출이 없으면 LLM이 결론을 내렸다는 것이므로 출력
        if not tool_called:
            print(f"\nFinal Answer: {ai_message.content}")
            return ai_message.content
        
        # 하나의 도구 사용만 강제하기
        tool_call = tool_called[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})
        tool_call_id = tool_call.get("id")

        print(f"    [Tool selected] {tool_name} with args : {tool_args}")

        tool_to_use = tool_dict.get(tool_name)
        if tool_to_use is None:
            raise ValueError(f"Tool '{tool_name}' not found")
        observation = tool_to_use.invoke(tool_args)

        print(f"    [Tool result] {observation}")

        # 반복 결과를 agent에게 더해줌
        # ToolMessage : LLM 에게 도구 사용 결과라고 알리기 위한 규격. id와 같이 전달 필수
        messages.append(ai_message)
        messages.append(
            ToolMessage(content=str(observation), tool_call_id=tool_call_id)
        )
    print("ERROR: Max iterations reached without a final answer")
    return None


if __name__ == "__main__":
    print("hello agent. (.bind_tools)")
    result = run_agent("What is the price of a laptop after applying a gold tier discount?")