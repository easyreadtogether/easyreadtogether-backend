import json
import argparse
import textstat

# from model import generate_easy_read, load_model
from model_aws import generate_easy_read

# more:
# https://en.wikipedia.org/wiki/Flesch%E2%80%93Kincaid_readability_tests
# https://readable.com/readability/smog-index/

"""
Flesch Reading Ease:
        Score	        School level	    Notes
    •	100.00-90.00	5th grade	        Very easy to read. Easily understood by an average 11-year-old student.
    •	90.0–80.0	    6th grade	        Easy to read. Conversational English for consumers.
    •	80.0–70.0	    7th grade	        Fairly easy to read.
    •	70.0–60.0	    8th & 9th grade	    Plain English. Easily understood by 13- to 15-year-old students.
    •	60.0–50.0	    10th to 12th grade	Fairly hard to read.
    •	50.0–30.0	    College	            Hard to read.
    •	30.0–0.0	    College graduate	Very hard to read. Best understood by university graduates.

Flesch–Kincaid readability tests:
a score of 9.3 means that a ninth grader would be able to read the document.
        Score	        School level (US)	Notes
    •	100.00–90.00	5th grade	        Very easy to read. Easily understood by an average 11-year-old student.
    •	90.0–80.0	    6th grade	        Easy to read. Conversational English for consumers.
    •	80.0–70.0	    7th grade	        Fairly easy to read.
    •	70.0–60.0	    8th & 9th grade	    Plain English. Easily understood by 13- to 15-year-old students.
    •	60.0–50.0	    10th to 12th grade	Fairly difficult to read.
    •	50.0–30.0	    College	            Difficult to read.
    •	30.0–10.0	    College graduate	Very difficult to read. Best understood by university graduates.
    •	10.0–0.0	    Professional	    Extremely difficult to read. Best understood by university graduates.


SMOG Index:
- Texts of fewer than 30 sentences are statistically invalid, because the SMOG formula was normed on 30-sentence samples. 
- textstat requires at least 3 sentences for a result.
- commonly used to assess the readability of *health-related materials* or other documents where clear communication is critical

	•	SMOG Grade 6-7: Easy to read; suitable for middle school students.
	•	SMOG Grade 8-12: Moderate difficulty; suitable for high school students.
	•	SMOG Grade 13+: Complex and challenging; suitable for college graduates or advanced readers.

Dale–Chall readability formula:
    •	4.9 or lower	average 4th-grade student or lower
    •	5.0–5.9	        average 5th- or 6th-grade student
    •	6.0–6.9	        average 7th- or 8th-grade student
    •	7.0–7.9	        average 9th- or 10th-grade student
    •	8.0–8.9	        average 11th- or 12th-grade student
    •	9.0–9.9	        average college student

McAlpine EFLAW Score:
https://strainindex.wordpress.com/2009/04/30/mcalpine-eflaw-readability-score/
- Returns a score for the readability of an english text for a foreign learner or English, focusing on the number of miniwords and length of sentences.
- It is recommended to aim for a score equal to or lower than 25.

    •	1-20    very easy to understand
    •	21-25   quite easy to understand
    •	26-29   a little difficult
    •	30+     very confusing
"""


def evaluate(text):
    # # [0-100] - higher score is easier to read
    # 100.00–90.00	5th grade	Very easy to read. Easily understood by an average 11-year-old student.
    flesch_reading_ease_score = textstat.flesch_reading_ease(text)
    flesch_kincaid_grade = textstat.flesch_kincaid_grade(
        text
    )  # [0-100] - lower score is easier to read
    smog_index = textstat.smog_index(text)
    dale_chall_readability_score = textstat.dale_chall_readability_score(text)
    mcalpine_eflaw_score = textstat.mcalpine_eflaw(text)

    return {
        "flesch_reading_ease_score": flesch_reading_ease_score,
        "flesch_kincaid_grade": flesch_kincaid_grade,
        "smog_index": smog_index,
        "dale_chall_readability_score": dale_chall_readability_score,
        "mcalpine_eflaw_score": mcalpine_eflaw_score,
    }


def main():
    """
    Evaluate the readability of a text

    Example:
    python evaluate.py \
        -f "./data/0 - Kampala Declaration on Jobs, Livelihoods & Self-reliance for Refugees, Returnees & Host Communities in IGAD Region.txt" \
        -m "meta.llama3-3-70b-instruct-v1:0" \
        -p "./prompts/prompt_1.txt" \
        -o "./output/0 - Kampala Declaration on Jobs, Livelihoods & Self-reliance for Refugees, Returnees & Host Communities in IGAD Region_easyread-prompt-3-Llama-3.3-70B.md"
    """
    parser = argparse.ArgumentParser(description="EasyRead Evaluation")

    parser.add_argument(
        "-f",
        "--file_path",
        type=str,
        required=True,
        help="File path to the text document",
    )
    parser.add_argument("-m", "--model_id", type=str, default=None, help="Model Id")
    parser.add_argument(
        "-p",
        "--prompt",
        type=str,
        default=None,
        help="Prompt filepath",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output filepath to save the generated easy to read content",
    )

    args = parser.parse_args()

    with open(args.file_path, "r") as f:
        text = f.read()

    # prompt
    if args.prompt:
        with open(args.prompt, "r") as f:
            system_prompt = f.read()

    if args.model_id:
        # model_pipeline, tokenizer, terminators = load_model(args.model_id)
        output = generate_easy_read(
            text,
            # model_pipeline=model_pipeline,
            # tokenizer=tokenizer,
            # terminators=terminators,
            model_id=args.model_id,
            system_prompt=system_prompt if args.prompt else None,
        )

        eval_results = evaluate(output)
        print(eval_results)

        if args.output:
            with open(args.output, "w") as f:
                f.write(output)

            # write evaluation results to file
            with open(args.output + ".json", "w") as f:
                json.dump(eval_results, f)
    else:
        eval_results = evaluate(text)
        print(eval_results)


if __name__ == "__main__":
    main()
