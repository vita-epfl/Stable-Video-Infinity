import re

def process_video_prompt(prompt_text, return_emotion=False):

    think_match = re.search(r'<think>(.*?)</think>', prompt_text, flags=re.DOTALL)
    think_content = think_match.group(1).strip() if think_match else ""
    
    answer_match = re.search(r'<answer>(.*?)</answer>', prompt_text, flags=re.DOTALL)
    answer_word = answer_match.group(1).strip() if answer_match else ""
    
    if think_content and answer_word:
        result = f"{think_content}\n\n***The person feels {answer_word}***"
    elif answer_word:
        result = f"***The person feels {answer_word} when talking***"
    else:
        result = re.sub(r'<[^>]*>', '', prompt_text)
    if return_emotion:
        return result, f"***The person feels {answer_word} when talking***"
    else:
        return result



def process_video_prompt_inference(prompt_text, return_emotion=False):

    think_match = re.search(r'<think>(.*?)</think>', prompt_text, flags=re.DOTALL)
    think_content = think_match.group(1).strip() if think_match else ""
    
    answer_match = re.search(r'<answer>(.*?)</answer>', prompt_text, flags=re.DOTALL)
    answer_word = answer_match.group(1).strip() if answer_match else ""
    

    result = f"***The person feels {answer_word} when talking***"

    if return_emotion:
        return result, f"{answer_word}"
    else:
        return result
