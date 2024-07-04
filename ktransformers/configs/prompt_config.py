
FIRST_TOKEN_WAITING_TIPS= "模型正在计算中，请稍等... "
FIRST_TOKEN_COMPLETE_TIPS= "模型Prefill计算完毕，共用时 "
#KVC2_FIRST_TOKEN_TIPS = "<span style='font-size: 20px; font-weight=bold'>KVC2</span>: First token response time is **:red[{}s]**"
#KVC2_FIRST_TOKEN_TIPS = '''<div style="margin-bottom:10px;"><b  style="font-size:20px;">KVC2</b>: First token response time is <b style="color:red;font-size:20px">{}s</b></div>'''
NO_KVC2_FIRST_TOKEN_TIPS = "Without KVC2: First token response time is {}s"
#NO_KVC2_FIRST_TOKEN_TIPS = '''<div style="margin-bottom:10px;"><b  style="font-size:20px;">Without KVC2</b>: First token response time is <b style="color:red;font-size:20px">{}s</b></div>'''

# cur_query_prompt = '<|im_start|>user\n上文是小说《三体》中部分章节的参考材料内容，\
#     材料中提供了多个《三体》的段落，部分材料与问题可能并不相关，需要你进行甄别，请根据材料内容回答符号###中的用户的提问。不需要你回答参考材料，只需要你回答问题，不要多说。\
#         ###{user}###<|im_end|>\n\
#     <|im_start|>assistant\n'
cur_query_prompt = '<|im_start|>user\n上文是小说《三体》中部分章节的参考材料内容，\
    材料中提供了多个《三体》的段落，段落之间可能不相关，部分材料与问题可能并不相关，需要你进行甄别，请根据材料内容回答符号###中的用户的提问。不需要你回答参考材料，只需要你回答问题，不要多说,不要说和问题无关的内容。现在请你回答下面的问题：\
        ###{user}###<|im_end|>\n\
    <|im_start|>assistant\n'
user_prompt = '<|im_start|>user\n{user}<|im_end|>\n'
robot_prompt = '<|im_start|>assistant\n{robot}<|im_end|>\n'