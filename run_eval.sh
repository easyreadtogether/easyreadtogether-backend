#!/bin/bash

# SUPPORTED_MODELS [
#     "meta.llama3-2-1b-instruct-v1:0",
#     "meta.llama3-2-3b-instruct-v1:0",
#     "meta.llama3-1-8b-instruct-v1:0",
#     "meta.llama3-3-70b-instruct-v1:0",
# ]

model_list=(
    "meta.llama3-2-1b-instruct-v1:0"
    "meta.llama3-2-3b-instruct-v1:0"
    "meta.llama3-1-8b-instruct-v1:0"
    "meta.llama3-3-70b-instruct-v1:0"
)

files_list=(
    # "./data/1 - Djibouti_Plan_Action_Education_2017_En.txt"
    # "./data/0 - Kampala Declaration on Jobs, Livelihoods & Self-reliance for Refugees, Returnees & Host Communities in IGAD Region.txt"
    "./data/2 - how-to-use-menstrual-hygiene-products.txt"
    "./data/3 - sexual-and-reproductive-rights.txt"
)

prompts_list=(
    "./prompts/prompt_1.txt"
    "./prompts/prompt_2.txt"
    "./prompts/prompt_3.txt"
)

output_folder="./output_2"

for model in "${model_list[@]}"
do
    model_safe="${model//:/_}"
    mkdir -p "$output_folder/$model_safe"

    for file in "${files_list[@]}"
    do
        for prompt in "${prompts_list[@]}"
        do
            echo "Evaluating $file with $model and $prompt"
            
            output_file="$output_folder/$model_safe/$(basename "$file")_$(basename "$prompt")_$(basename "$model_safe").txt"

            python evaluate.py \
                -f "$file" \
                -m "$model" \
                -p "$prompt" \
                -o "$output_file"
            
            echo "---------------------------------------------------------------------------------------------------------"
        done
    done
done


# python evaluate.py \
#     -f "./data/0 - Kampala Declaration on Jobs, Livelihoods & Self-reliance for Refugees, Returnees & Host Communities in IGAD Region.txt" \
#     -m "meta.llama3-3-70b-instruct-v1:0" \
#     -p "./prompts/prompt_1.txt" \
#     -o "./output/0 - Kampala Declaration on Jobs, Livelihoods & Self-reliance for Refugees, Returnees & Host Communities in IGAD Region_easyread-prompt-3-Llama-3.3-70B.md"


## ChatGPT

python evaluate.py \
    -f "./output_2/ChatGPT/0 - Kampala Declaration on Jobs, Livelihoods & Self-reliance for Refugees, Returnees & Host Communities in IGAD Region.txt_prompt_2.txt_ChatGPT.txt"
