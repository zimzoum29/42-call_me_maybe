*This project has been created as part of the 42 curriculum by tigondra.*

# Call Me Maybe

## Description

Call Me Maybe is a function-calling system for local language models. It reads natural-language prompts and function definitions, then outputs structured function calls in JSON:

```json
{
  "prompt": "What is the sum of 2 and 3?",
  "name": "fn_add_numbers",
  "parameters": {"a": 2.0, "b": 3.0}
}
```

The program does not ask the model to freely write JSON and hope that the result is valid. Instead, it uses constrained decoding: every next token proposed by the model is checked against a JSON grammar before it can be selected.

## Instructions

Install dependencies:

```bash
make install
```

Run with default paths:

```bash
make run
```

Run with explicit paths:

```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json
```

Clean generated files:

```bash
make clean
```

Debug Mode:

```bash
make debug
```

Lint:

```bash
make lint
```

Lint Strict:

```bash
make lint-strict
```

## Algorithm explanation

The core of the project is a token-by-token constrained decoder.

For each prompt, the model receives the user request and the list of available functions. The model still chooses the function and the arguments through its logits, so function selection is not hardcoded with heuristics.

At every generation step:

1. The current prompt plus already generated tokens are sent to the LLM.
2. The LLM returns logits for the next token over the whole vocabulary.
3. The decoder tests candidate tokens by decoding each token id into text and appending it to the current output prefix.
4. The resulting prefix must still match at least one valid branch of the grammar.
5. The highest-logit valid token is appended.
6. Generation stops only when the complete output is a valid schema-compliant JSON object.

The grammar is not only a JSON checker. It also encodes the expected schema:

```json
{"name":"<existing_function_name>","parameters":{"arg":value}}
```

For example, while generating the `name` field, the decoder only accepts tokens that continue one of the real function names from `functions_definition.json`. When generating parameters, it enforces the exact parameter names, order, and types declared by the selected function branch.

Supported parameter types are:

- `string`
- `number`
- `integer`
- `boolean`

A final Pydantic validation step is kept as a safety check, but the output should already be valid before that step.

## Design decisions

The project uses compact JSON without spaces. This keeps the grammar small and makes token-level validation simpler and faster.

Each function definition becomes one grammar branch. The global grammar is the union of all branches, so the model can choose any available function, but only valid function names and schemas are possible.

The decoder checks the highest-logit tokens first for performance. If none of those tokens is valid, it scans the rest of the vocabulary. This keeps the behavior equivalent to masked decoding while avoiding unnecessary work in common cases.

The code avoids private attributes of `llm_sdk` and only uses the public methods required by the subject: encoding, decoding, and next-token logits.

The default model is `Qwen/Qwen3-0.6B`, but you can use `cyberbabooshka/base_noreasoning` to demonstrate that the project works with a different model

## Performance analysis

Reliability is high for JSON validity because invalid syntax cannot be generated. The output file is always written with `json.dump`, and generated calls are parsed and validated before being added to the final result.

Accuracy depends on the small model's logits, but constrained decoding improves reliability because the model chooses only among legal function-call continuations. This removes malformed JSON, extra prose, missing keys, wrong parameter names, and wrong primitive JSON types.

Speed is kept reasonable by checking the most likely tokens first and falling back to a full vocabulary scan only when needed.

## Challenges faced

The main challenge is that tokenizers do not generate one character at a time. A single token can contain several characters, punctuation, or part of a word. The grammar therefore validates full text prefixes after appending a decoded token, not just individual characters.

Another challenge is distinguishing real constrained decoding from post-processing. This implementation does not repair malformed model output. It prevents invalid output during generation.

## Testing strategy

Recommended tests:

- Valid function definitions with string, number, integer, and boolean parameters.
- Functions with zero, one, and multiple parameters.
- Prompts containing quoted strings and special characters.
- Invalid input JSON files.
- Missing input files.
- Ambiguous prompts where the model must choose between similar functions.
- Output validation with `json.load` and Pydantic models.

## Example usage

Input prompt:

```json
{"prompt": "Reverse the string 'hello'"}
```

Function definition:

```json
{
  "name": "fn_reverse_string",
  "description": "Reverse a string and return the reversed result.",
  "parameters": {
    "s": {"type": "string"}
  },
  "returns": {"type": "string"}
}
```

Expected output shape:

```json
{
  "prompt": "Reverse the string 'hello'",
  "name": "fn_reverse_string",
  "parameters": {"s": "hello"}
}
```

## Resources

- JSON specification and Python `json` module documentation.
- Pydantic documentation for data validation.
- The provided `llm_sdk` wrapper.
- General documentation on constrained decoding and structured generation.

AI was used to review the project architecture, understand what is clearly constrained decoding, and help to test effectively the program. The final code should be understood, tested, and defended by the authors.
