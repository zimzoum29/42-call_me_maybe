import argparse
import json
from typing import Any, TypedDict

from src.decoder import ConstrainedDecoder
from src.grammar import Grammar, normalize_types
from src.llm_engine import FunctionCallingEngine
from src.models import (
    FunctionCall,
    validate_function_definitions,
    validate_prompt_items,
)
from src.utils import JsonError, read_json, write_json


class CLIArgs(TypedDict):
    functions_definition: str
    input: str
    output: str
    visualize: bool


def parse_args() -> CLIArgs:
    parser = argparse.ArgumentParser(
        description="Generate schema-valid function calls from prompts."
    )
    parser.add_argument(
        "--functions_definition",
        default="data/input/functions_definition.json",
        help="Path to the function definitions JSON file.",
    )
    parser.add_argument(
        "--input",
        default="data/input/function_calling_tests.json",
        help="Path to the prompt tests JSON file.",
    )
    parser.add_argument(
        "--output",
        default="data/output/function_calling_results.json",
        help="Path where generated results will be written.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Display live decoding in the terminal.",
    )
    args = parser.parse_args()
    return {
        "functions_definition": args.functions_definition,
        "input": args.input,
        "output": args.output,
        "visualize": args.visualize,
    }


def decode_prompt(
    decoder: ConstrainedDecoder,
    engine: FunctionCallingEngine,
    user_prompt: str,
    visualize: bool,
) -> dict[str, Any]:
    if visualize is True:
        print()
        print("Prompt:", user_prompt)
    prompt_text = engine.build_prompt(user_prompt)
    generated = decoder.generate(prompt_text)
    try:
        raw_call = json.loads(generated.text)
        call = FunctionCall.model_validate(raw_call)
    except Exception as error:
        raise JsonError(
            "Generated invalid JSON despite constraints: "
            f"{error}"
        ) from error
    function = engine.get_function_definition(call.name)
    parameters = normalize_types(call.parameters, function)
    if visualize is True:
        print({"name": call.name, "parameters": parameters})
    return {
        "prompt": user_prompt,
        "name": call.name,
        "parameters": parameters,
    }


def run(
    functions_definition_path: str,
    input_path: str,
    output_path: str,
    visualize: bool,
) -> None:
    raw_functions = read_json(functions_definition_path)
    raw_prompts = read_json(input_path)
    functions = validate_function_definitions(raw_functions)
    prompts = validate_prompt_items(raw_prompts)

    print(f"Loaded {len(functions)} function definitions.")
    print(f"Loaded {len(prompts)} prompts.")

    engine = FunctionCallingEngine(functions)
    decoder = ConstrainedDecoder(engine, Grammar(functions))
    results = [decode_prompt(decoder, engine, item.prompt, visualize)
               for item in prompts]
    write_json(output_path, results)


def main() -> None:
    args = parse_args()
    try:
        run(
            args["functions_definition"],
            args["input"],
            args["output"],
            args["visualize"],
        )
        print("Program finished successfully.")
    except Exception as error:
        print(f"[ERROR] {error}")


if __name__ == "__main__":
    main()
