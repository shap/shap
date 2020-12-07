''' Tests for Explainer class.
'''

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
    wrapped_model = TeacherForcingLogits(f,similarity_model=model, similarity_tokenizer=tokenizer)
    masker = maskers.Text(tokenizer, mask_token = "...")
    
    explainer = shap.Explainer(wrapped_model,masker)

    assert safe_isinstance(explainer.masker, "shap.maskers.FixedComposite")