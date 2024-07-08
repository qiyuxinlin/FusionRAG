# Server
<img src="server-arch.png" height="600" alt="Server架构">

## API

Server 通过 RESTful API 对外提供模型推理服务，提供 Chat Completion 和 Assistant 两种调用方式。

- ChatCompletion 接口要求用户一次提供所有的历史对话，然后把模型推理的结果返回。对于 Chat Completion，Serve 提供和  [Ollama](https://github.com/ollama/ollama/blob/main/docs/api.md)  和 [OpenAI](https://platform.openai.com/docs/api-reference/chat/create)  一致的 API 接口。因此当前应用可以无缝切换到我们的 Server。例如： [如何使用 tabby 和 ktransformers 在本地做代码补全？]()。
- 而对于 Assistant 方式，首先需要创建一个包含初始命令、相关文件和相关对话（Related Threads）的助理（Assistant），然后应用将 Server 将用户的消息（Message）存储在对话（Thread）中，之后创建一次运行（Run）来调用模型获取回复。Server 提供和  [OpenAI Assistant API](https://platform.openai.com/docs/api-reference/assistants/createAssistant) 一致的 API 接口，并计划支持 OpenAI SDK。

## Inference Framework

Server 通过 ktransformers 调用模型并进行推理，Server 也支持其他的推理框架，例如已经支持的 [transformers](https://huggingface.co/docs/transformers/index) ，并计划支持 [exllamav2](https://github.com/turboderp/exllamav2)。这些功能主要由图中的`server/backend` 完成。

Server 将推理框架的推理功能抽象成一个异步的generator函数：inference。它的输入是是历史的对话信息 messages，输出是模型返回的文字结果。

```python
  async def inference(self, messages, request_unique_id:Optional[str],**kwargs)->AsyncIterator[str]:
```

为了实现 Assistant 的相关功能，Server 在 `server/schemas` 里定义了 pydantic object，通过`server/crud` 中的 CRUD 函数操作数据，并在 `server/models` 里通过 ORM 和 sqlite 数据库进行同步。此外，Server 还需要实现一个 ThreadContext，以和 BackendInterface 进行对接。get_local_messages是需要实现的函数，其主要功能是将存储在 Thread 中的对话信息提取出来，并格式化成 BackendInterface 能够接受的形式。ThreadContext和BackendInterfaceBase都定义在 `server/backend/base.py`。

```python
class MyThreadContext(ThreadContext):
    def get_local_messages(self):
        '''
        Get local messages, as the input to interface.work
        This function is intended to message preprocess e.g. apply chat template
        '''
        # implmentations
```









### Chat Completions 

Chat Completion 可以根据上传的历史对话传给模型，补全之后返回给用户。例如：

```bash
curl -X 'POST' \
  'http://localhost:9112/v1/chat/completions' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "messages": [
    {
      "content": "tell a joke",
      "role": "user"
    }
  ],
  "model": "Llama",
  "stream": false
}'
```

```json
{
  "id": "66e7e3af-322a-4bb1-90ed-a8e9b32edb2a",
  "object": "chat.completion.chunk",
  "created": 1720433138,
  "model": "not implmented",
  "system_fingerprint": "not implmented",
  "usage": null,
  "choices": [
    {
      "index": 0,
      "message": {
        "content": "Here's one:\n\nWhy couldn't the bicycle stand up by itself?\n\n(wait for it...)\n\nBecause it was two-tired!\n\nHope that made you smile! Do you want to hear another one?",
        "role": "assistant",
        "name": null
      },
      "logprobs": null,
      "finish_reason": null
    }
  ]
}
```



