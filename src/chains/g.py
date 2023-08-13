# Chains


from langchain import LLMChain, PromptTemplate
from langchain.chains import SequentialChain, MultiPromptChain
from langchain.chains.router.llm_router import RouterOutputParser, LLMRouterChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import src._dotenv

product_name_template = """
You are a naming consultant for new companies. Given a product name, you come up with great and funny product names!

Here is a product name:
{input}
"""

translation_template = """
You are a brilliant translator to translate from English to German and French. Given an input you translate it into German and French.

Here is some input:
{input}
"""


prompt_infos = [
    {
        "name": "product_name",
        "description": "Good to create creative product names",
        "prompt_template": product_name_template
    },
    {
        "name": "translate",
        "description": "Good for translating texts",
        "prompt_template": translation_template
    }
]

llm = ChatOpenAI(temperature=0.9)

destination_chains = {}
for p_info in prompt_infos:
    name = p_info["name"]
    prompt_template = p_info["prompt_template"]
    prompt = ChatPromptTemplate.from_template(template=prompt_template)
    chain = LLMChain(llm=llm, prompt=prompt)
    destination_chains[name] = chain

destinations = [f"{p['name']}: {p['description']}" for p in prompt_infos]
destinations_str = "\n".join(destinations)

default_prompt = ChatPromptTemplate.from_template("{input}")
default_chain = LLMChain(llm=llm, prompt=default_prompt)


MULTI_PROMPT_ROUTER_TEMPLATE = """Given a raw text input to a \
language model select the model prompt best suited for the input. \
You will be given the names of the available prompts and a \
description of what the prompt is best suited for. \
You may also revise the original input if you think that revising\
it will ultimately lead to a better response from the language model.

<< FORMATTING >>
Return a markdown code snippet with a JSON object formatted to look like:
```json
{{{{
    "destination": string \ name of the prompt to use or "DEFAULT"
    "next_inputs": string \ a potentially modified version of the original input
}}}}
```

REMEMBER: "destination" MUST be one of the candidate prompt \
names specified below OR it can be "DEFAULT" if the input is not\
well suited for any of the candidate prompts.
REMEMBER: "next_inputs" can just be the original input \
if you don't think any modifications are needed.

<< CANDIDATE PROMPTS >>
{destinations}

<< INPUT >>
{{input}}

<< OUTPUT (remember to include the ```json)>>"""


router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(
    destinations=destinations_str
)
router_prompt = PromptTemplate(
    template=router_template,
    input_variables=["input"],
    output_parser=RouterOutputParser(),
)

router_chain = LLMRouterChain.from_llm(llm, router_prompt)


chain = MultiPromptChain(router_chain=router_chain,
                         destination_chains=destination_chains,
                         default_chain=default_chain, verbose=True
                        )


#response = chain.run("Hey there, I produce colorful socks, what should my company be named?")
response = chain.run("Hey there, my company 'Socktastic' is starting in Europe, can you please give me the French and German names for that?")
print(response)





