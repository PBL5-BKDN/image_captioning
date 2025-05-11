curl https://api.x.ai/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer xai-w7o4GJrhDcZ8uv1jw8hpmyq1bHUHPPZTg35dpJJut77P8fAKgTvMrt1RrYG4GMrl6cfI2fg5Wd63m8by" \
  -d '{
  "messages": [
    {
      "role": "system",
      "content": "You are a test assistant."
    },
    {
      "role": "user",
      "content": "Testing. Just say hi and hello world and nothing else."
    }
  ],
  "model": "grok-3-latest",
  "stream": false,
  "temperature": 0
}'