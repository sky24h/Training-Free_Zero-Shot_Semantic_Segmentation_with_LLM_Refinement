import torch
import signal
from openai import OpenAI
from termcolor import colored

import transformers
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer

# environment variables and paths
from .env_utils import get_device, low_vram_mode

device = get_device()

class GPT:
    def __init__(self, model="gpt-4o-mini", api_key=None):
        self.prices = {
            # check at https://openai.com/api/pricing/
            "gpt-3.5-turbo-0125": [0.0000005, 0.0000015],
            "gpt-4o-mini"       : [0.00000015, 0.00000060],
            "gpt-4-1106-preview": [0.00001, 0.00003],
            "gpt-4-0125-preview": [0.00001, 0.00003],
            "gpt-4-turbo"       : [0.00001, 0.00003],
            "gpt-4o"            : [0.000005, 0.000015],
        }
        self.cheaper_model = "gpt-4o-mini"
        assert model in self.prices.keys(), "Invalid model, please choose from: {}, or add new models in the code.".format(self.prices.keys())
        self.model = model
        print(f"Using {model}")
        self.client = OpenAI(api_key=api_key)
        self.total_cost = 0.0

    def _update(self, response, price):
        current_cost = response.usage.completion_tokens * price[0] + response.usage.prompt_tokens * price[1]
        self.total_cost += current_cost
        # print in 4 decimal places
        print(
            colored(
                f"Current Tokens: {response.usage.completion_tokens + response.usage.prompt_tokens:d} \
                Current cost: {current_cost:.4f} $, \
                Total cost: {self.total_cost:.4f} $",
                "yellow",
            )
        )

    def chat(self, messages, temperature=0.0, max_tokens=200, post=False):
        # set temperature to 0.0 for more deterministic results
        if post:
            # use cheaper model for post-refinement to save costs, since the task is simpler.
            generated_text = self.client.chat.completions.create(
                model=self.cheaper_model, messages=messages, temperature=temperature, max_tokens=max_tokens
            )
            self._update(generated_text, self.prices[self.cheaper_model])
        else:
            generated_text = self.client.chat.completions.create(
                model=self.model, messages=messages, temperature=temperature, max_tokens=max_tokens
            )
            self._update(generated_text, self.prices[self.model])
        generated_text = generated_text.choices[0].message.content
        return generated_text


class Llama3:
    def __init__(self, model="Meta-Llama-3-8B-Instruct"):
        model = "meta-llama/{}".format(model)  # or replace with your local model path
        print(f"Using {model}")
        tokenizer = AutoTokenizer.from_pretrained(model)
        if low_vram_mode:
            model = AutoModelForCausalLM.from_pretrained(
                model, quantization_config=BitsAndBytesConfig(load_in_8bit=True), device_map="auto"
            ).eval()
        self.pipeline = transformers.pipeline(
            "text-generation",
            model        = model,
            tokenizer    = tokenizer,
            model_kwargs = {"torch_dtype": torch.float16},
            device_map   = "auto",
        )
        self.terminators = [self.pipeline.tokenizer.eos_token_id, self.pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")]

    def _update(self):
        print(colored("Using Llama-3, Free", "green"))

    def chat(self, messages, temperature=0.0, max_tokens=200, post=False):
        prompt = self.pipeline.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        generated_text = self.pipeline(
            prompt,
            max_new_tokens = max_tokens,
            eos_token_id   = self.terminators,
            pad_token_id   = 128001,
            do_sample      = True,
            temperature    = max(temperature, 0.01), # 0.0 is not supported
            top_p          = 0.9,
        )
        self._update()
        generated_text = generated_text[0]["generated_text"][len(prompt) :]
        return generated_text


# Define the timeout handler
def timeout_handler(signum, frame):
    raise TimeoutError()


def init_model(model, api_key=None):
    if "gpt" in model:
        return GPT(model=model, api_key=api_key)
    elif "Llama" in model:
        return Llama3(model=model)
    else:
        raise ValueError("Invalid model")


def _generate_example_prompt(examples, llm=None):
    # system prompt
    system_prompt = """
    Task Description:
    - you will provide detailed explanations for example inputs and outputs within the context of the task.

    Please adhere to the following rules:
    - Exclude terms that appear in both lists.
    - Detail the relevance of unmatched terms from input to output, focusing on indirect relationships.
    - Identify and explain terms common to all output lists but rarely present in input lists; include these at the end of the output labeled 'Recommend Include Labels'.
    - Each explanation should be concise, around 50 words.

    Output Format:
    - '1. Input... Output... Explanation... n. Input... Output... Explanation... \n Recommend Include Labels: label1, labeln, ...'
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"Here are the input and output lists for which you need to provide detailed explanations:{examples.strip()}",
        },
    ]
    generated_example = llm.chat(messages, temperature=0.0, max_tokens=1000)
    return generated_example


def _make_prompt(label_list, example=None):
    Cityscape = "sidewalk" in label_list
    if Cityscape:
        add_text = f'contain at least {len(label_list.split(", "))} labels, '
    else:
        add_text = ""
    # Task description and instructions for processing the input to generate output
    system_prompt = f"""
    Task Description:
    - You will receive a list of caption tags accompanied by a caption text and must assign appropriate labels from a predefined label list: "{label_list}".

    Instructions:
    Step 1. Visualize the scene suggested by the input caption tags and text.
    Step 2. Analyze each term within the overall scene to predict relevant labels from the predefined list, ensuring no term is overlooked.
    Step 3. Now forget the input list and focus on the scene as a whole, expanding upon the labels to include any contextually relevant labels that complete the scene or setting.
    Step 4. Compile all identified labels into a comma-separated list, adhering strictly to the specified format.

    Contextually Relevant Tips:
    - Equivalencies include converting "girl, man" to "person" and "flower, vase" to "potted plant", while "bicycle, motorcycle" suggest "rider".
    - An outdoor scene may include labels like "sky", "tree", "clouds", "terrain".
    - An urban scene may imply "bus", "bicycle", "road", "sidewalk", "building", "pole", "traffic-light", "traffic-sign".

    Output:
    - Do not output any explanations other than the final label list.
    - The final output should {add_text}strictly adhere to the specified format: label1, label2, ... labeln
    """.strip()
    if example:
        system_prompt += f"""
        Additional Examples with Detailed Explanations:
        {example}
        """
    print("system_prompt: ", system_prompt)
    return system_prompt

    # - You will receive a list of terms accompanied by a caption text and must assign appropriate labels from a predefined label list: "{label_list}".

    # Instructions:
    # Step 1. Visualize the scene suggested by the input list and caption text.


def make_prompt(label_list, example_rams=None, example_gts=None, llm=None):
    # Create a string representation of the examples
    if example_rams is None or example_gts is None:
        generated_example = None
    else:
        examples = ""
        for n, (example_ram, example_gt) in enumerate(zip(example_rams, example_gts)):
            examples += f"{n+1}. Input: {example_ram}, Output: {example_gt} "

        # Generate an example prompt using the string representation of the examples
        generated_example = _generate_example_prompt(examples, llm=llm)
        print("generated_example: ", generated_example)

    # Create a new system prompt using the label list and the improved example prompt
    system_prompt = _make_prompt(label_list, generated_example)
    system_prompt = {"role": "system", "content": system_prompt.strip()}
    print("system_prompt: ", system_prompt)
    return system_prompt


def _call_llm(system_prompt, llm, user_input):
    # Set the signal handler and a 10-second alarm
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(10)
    try:
        messages = [system_prompt, {"role": "user", "content": "Here are input caption tags and text: " + user_input}]
        converted_label = llm.chat(messages=messages, temperature=0.0, max_tokens=200)
        return converted_label
    except TimeoutError:
        return ""
    finally:
        # Disable the alarm
        signal.alarm(0)


def pre_refinement(user_input_list, system_prompt, llm=None):
    llm_outputs = [_call_llm(system_prompt, llm, user_input) for user_input in user_input_list]
    converted_labels = [f"{user_input_}, {converted_label}" for user_input_, converted_label in zip(user_input_list, llm_outputs)]
    return converted_labels, llm_outputs


def post_refinement(label_list, detected_label, llm=None):
    system_input = f"""
    Task Description:
    - You will receive a specific phrase and must assign an appropriate label from the predefined label list: "{label_list}". \n \

    Please adhere to the following rules: \n \
    - Select and return only one relevant label from the predefined label list that corresponds to the given phrase. \n \
    - Do not include any additional information or context beyond the label itself. \n \
    - Format is purely the label itself, without any additional punctuation or formatting. \n \
    """
    system_input = {"role": "system", "content": system_input}
    messages = [system_input, {"role": "user", "content": detected_label}]
    if detected_label == "":
        return ""
    generated_label = None
    for count in range(3):
        generated_label = llm.chat(messages=messages, temperature=0.0 if count == 0 else 0.1 * (count), post=True)
        if generated_label != "":
            break
    return generated_label


if __name__ == "__main__":
    # test the functions
    llm = Llama3(model="Meta-Llama-3-8B-Instruct")

    system_prompt = make_prompt("person, car, tree, sky, road, building, sidewalk, traffic-light, traffic-sign", llm=llm)

    converted_labels, llm_outputs = pre_refinement(["person, car, road, traffic-light"], system_prompt, llm=llm)
    print("converted_labels: ", converted_labels)
    print("llm_outputs: ", llm_outputs)