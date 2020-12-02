''' Tests for Explainer class.
'''

def test_wrapping_for_text_to_text_pretrained_transformer_model():
    import shap
    from shap.utils import safe_isinstance
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model =  AutoModelForCausalLM.from_pretrained("gpt2")

    explainer = shap.Explainer(model,tokenizer)

    assert safe_isinstance(explainer.elemental_model, "shap.models.TeacherForcingLogits") \
            and safe_isinstance(explainer.masker, "shap.maskers.FixedComposite")

def test_wrapping_for_text_to_text_teacher_forcing_logits_model():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import shap
    from shap.utils import safe_isinstance
    from shap.models import TeacherForcingLogits
    from shap import maskers
    
    def f(x):
        pass

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model =  AutoModelForCausalLM.from_pretrained("gpt2")
    wrapped_model = TeacherForcingLogits(f,text_similarity_model=model, text_similarity_tokenizer=tokenizer)
    masker = maskers.Text(tokenizer, mask_token = "<infill>")
    
    explainer = shap.Explainer(wrapped_model,masker)

    assert safe_isinstance(explainer.masker, "shap.maskers.FixedComposite")