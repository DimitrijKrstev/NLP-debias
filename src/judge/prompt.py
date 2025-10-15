def build_judge_prompt(biased_text: str, model_output: str, reference_text: str) -> str:

    return f"""You are an expert evaluator of bias-neutralization systems.
        Given a biased input, a model's neutralized output, and a human reference neutral text,
        evaluate the model output on five metrics. 
        Each score should be an integer from 1 (very poor) to 10 (excellent).

        Return your response as strict JSON with:
        - a numeric score for each metric, and
        - one concise "overall_reasoning" sentence summarizing the overall assessment.

        BIAS INPUT:
        {biased_text}

        MODEL OUTPUT:
        {model_output}

        REFERENCE (human neutral text):
        {reference_text}

        Metrics to score (1–10):
        1. Neutrality — How well does the text remove subjective or biased language?
        2. Meaning Preservation — Does it retain the factual meaning of the biased text?
        3. Fluency & Coherence — Is it grammatically correct and natural?
        4. Faithfulness to Reference — How similar is it to the reference neutral text in tone and meaning?
        5. Edit Minimality — Does it make only necessary changes to achieve neutrality?

        Return strict JSON in this format:
        {{
            "neutrality": 9,
            "meaning_preservation": 8,
            "fluency": 10,
            "faithfulness_to_reference": 9,
            "edit_minimality": 8,
            "overall_reasoning": "The rewrite effectively removes bias with minimal edits while preserving meaning and fluency close to the human reference."
        }}
        """
