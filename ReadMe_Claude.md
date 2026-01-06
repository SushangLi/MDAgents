system: macos
python: 3.14
project abstract: multi-agent systems, MAS, for scientific computing and plotting.
basic tasks:
0. read requirments.txt to get current available python packages.
1. use model context protocol, mcp, to build connections between ai agents.
2. build a main.py to run a interactive mode in console.
3. the MAS is dominated by a LLM like gemini-3, gpt-5, deepseek. It can call local tools to understand local file contents. then call other expert agents like nanobanana(the gemini-flash-image) to excute. 
4. all ai agents including the core LLM should be called with api keys which set in the local env files. 
