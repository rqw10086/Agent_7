import ast
import inspect
from collections.abc import Callable
from pathlib import Path

from openai import OpenAI

ToolFunction = Callable[..., str]
TOOL_REGISTRY: dict[str, ToolFunction] = {}


def register_tool(function: ToolFunction) -> ToolFunction:
    TOOL_REGISTRY[function.__name__] = function
    return function


def build_prompt(user_input: str) -> str:
    """根据工具函数的签名和文档字符串动态构建工具提示词。"""
    tool_descriptions = "\n".join(
        f"- {name}{inspect.signature(function)}: {inspect.getdoc(function)}"
        for name, function in TOOL_REGISTRY.items()
    )
    return f"""你是一个 ReAct 助手。根据需要调用以下工具：
{tool_descriptions}

需要调用工具时，每次只调用一个，并严格按以下格式回复，不要使用代码围栏：
Thought: 简要说明本步要做什么
Action: 工具函数名(参数名=Python字面量, ...)
ToolResult:

例如：
Action: query_family_tree(name="小明", relationship="哥哥")
ToolResult:

ToolResult: 是本轮工具调用的结束标记，输出它后立即停止，等待工具返回。
收到 ToolResult 后，可以继续调用其他工具。信息足够时直接回答，不要输出 Action 或 ToolResult。
如何你已经得到最终答案，请直接输出
Answer: 最终答案
不要再调用工具和附带工具调用相关关键词。

用户: {user_input}"""


def call_llm(prompt: str) -> str:
    # 默认会自动读取环境变量中的 OPENAI_API_KEY
    client = OpenAI()
    client.base_url = "http://192.168.124.18:18082/v1"
    response = client.chat.completions.create(
        model="qwen38-gyro-m",
        messages=[
            {"role": "system", "content": "你是一个乐于助人的助手。"},
            {"role": "user", "content": prompt},
        ],
        stop=["ToolResult:"],
    )
    content = response.choices[0].message.content
    if content is None:
        raise ValueError("模型没有返回文本内容")
    return content



def parse_action(response_text: str) -> tuple[str, dict[str, object]] | None:
    """解析 Action 行中的函数调用；ToolResult 由模型 API 的 stop 参数截断。"""
    for line in response_text.splitlines():
        key, separator, value = line.partition(":")
        if not separator or key != "Action":
            continue
        try:
            expression = ast.parse(value.strip(), mode="eval").body
        except SyntaxError as error:
            raise ValueError("Action 必须是函数调用表达式") from error
        if not isinstance(expression, ast.Call) or not isinstance(expression.func, ast.Name):
            raise ValueError("Action 必须是函数名及其参数")
        if expression.args or any(keyword.arg is None for keyword in expression.keywords):
            raise ValueError("Action 只支持显式命名参数")
        arguments: dict[str, object] = {}
        for keyword in expression.keywords:
            if keyword.arg is None:
                raise ValueError("Action 只支持显式命名参数")
            arguments[keyword.arg] = ast.literal_eval(keyword.value)
        return expression.func.id, arguments
    return None


FAMILY_TREE = {
    "小明": {
        "父亲": "小刚",
        "母亲": "小红",
        "哥哥": "小华",
    },
    "小刚": {
        "儿子": "小明",
        "妻子": "小红",
    },
    "小红": {
        "儿子": "小明",
        "丈夫": "小刚",
    },
    "小华": {
        "兄弟": "小明",
        "父亲": "小刚",
        "母亲": "小红",
    },
}


@register_tool
def query_family_tree(name: str, relationship: str) -> str:
    """查询亲属关系；提供 name（姓名）和 relationship（关系名称）。"""
    if name in FAMILY_TREE and relationship in FAMILY_TREE[name]:
        return FAMILY_TREE[name][relationship]
    return "未找到相关信息"



@register_tool
def read_hobbies_file() -> str:
    """读取家庭成员兴趣爱好文档；无参数，返回文档全文供后续回答使用。"""
    hobbies_path = Path(__file__).with_name("家庭成员兴趣爱好.md")
    if not hobbies_path.is_file():
        raise FileNotFoundError(f"兴趣爱好文档不存在: {hobbies_path}")
    return hobbies_path.read_text(encoding="utf-8")


@register_tool
def list_family_members() -> str:
    """列出族谱中的所有家庭成员；无参数。"""
    return "、".join(FAMILY_TREE)


@register_tool
def example_action(value: str) -> str:
    """示例工具：回显 value 参数，用于演示工具调用。"""
    return f"这是一个示例工具结果: {value}"


def run_tool(action: str, arguments: dict[str, object]) -> str:
    """按工具名称分发调用，并传入 Action 中携带的参数。"""
    function = TOOL_REGISTRY.get(action)
    if function is None:
        raise ValueError(f"未知的动作: {action}")
    try:
        inspect.signature(function).bind(**arguments)
    except TypeError as error:
        raise ValueError(f"{action} 参数无效: {error}") from error
    return function(**arguments)


def agent_loop(user_input: str, max_steps: int = 5) -> str:
    """循环执行模型提出的工具调用，直到得到最终答案或达到步数上限。"""
    if max_steps < 1:
        raise ValueError("max_steps 必须大于 0")

    prompt = build_prompt(user_input)
    for step in range(max_steps + 1):
        response_text = call_llm(prompt)
        try:
            tool_call = parse_action(response_text)
        except Exception as error:
            prompt += (
                f"\n\n{response_text}\n{error}\n"
                "请根据上述错误原文自行调整参数或上下文后重试。"
            )
            continue

        if tool_call is None:
            print("最终答案:", response_text)
            return response_text
        if step == max_steps:
            prompt += (
                f"\n\n{response_text}\n"
                f"单次对话的工具调用次数已达上限（{max_steps} 步），不能再调用任何工具。\n"
                "请基于以上已有的工具结果直接给出最终答案，不要再输出 Action 或 ToolResult。"
            )
            final_text = call_llm(prompt)
            print("最终答案:", final_text)
            return final_text

        action, arguments = tool_call
        try:
            observation = run_tool(action, arguments)
        except Exception as error:
            prompt += (
                f"\n\n{response_text}\n{error}\n"
                "请根据上述错误原文自行调整参数或上下文后重试。"
            )
            continue
        print(f"ToolResult: {observation}")
        prompt += (
            f"\n\n{response_text}\n"
            f"ToolResult: {observation}\n"
            "请基于以上工具结果继续完成任务；如需更多信息，按格式调用下一工具。"
        )

    prompt += (
        f"\n\n单次对话的工具调用次数已达上限（{max_steps} 步），不能再调用任何工具。\n"
        "请基于以上已有的工具结果直接给出最终答案，不要再输出 Action 或 ToolResult。"
    )
    final_text = call_llm(prompt)
    print("最终答案:", final_text)
    return final_text


if __name__ == "__main__":
    agent_loop("小明的哥哥是谁？他有什么兴趣爱好？")
