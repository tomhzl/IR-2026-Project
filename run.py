#!/usr/bin/env python3
"""Run a student solve() adapter or the clearly labeled offline format demo."""
from __future__ import annotations
import argparse
import importlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import signal
import sys
import time
from ir2026 import BudgetExceeded,Corpus,CorpusTools,DataError,PROTOCOL_VERSION,count_model_tool_calls,load_policy,load_questions,read_jsonl,write_jsonl

ROOT=Path(__file__).resolve().parent

def worker(connection,agent_name,question,corpus_root,config,trace_path):
    if hasattr(os,'setsid'):
        try:os.setsid()
        except OSError:pass
    started=time.monotonic();tools=None
    try:
        if config.get('api_provider')=='siliconflow':
            from agent.answer import SILICONFLOW_API_BASE
            os.environ['IR2026_API_BASE']=SILICONFLOW_API_BASE
        tools=CorpusTools(Corpus(corpus_root),question['id'],config['limits'],trace_path=trace_path,method_type=config['method_type'])
        module=importlib.import_module(agent_name)
        result=module.solve(dict(question),tools,config)
        if not isinstance(result,dict):raise DataError('solve() must return a dictionary')
    except BudgetExceeded as exc:
        result={'answer':'','doc_ids':[],'evidence':[],'status':'budget_exceeded','error':str(exc)}
    except Exception as exc:
        result={'answer':'','doc_ids':[],'evidence':[],'status':'error','error':f'{type(exc).__name__}: {exc}'}
    result['id']=question['id']
    usage=result.get('usage',{})
    if not isinstance(usage,dict):usage={}
    usage.update({'tool_calls':count_model_tool_calls(tools.trace) if tools else 0,
                  'trace_events':len(tools.trace) if tools else 0,'elapsed_seconds':round(time.monotonic()-started,4)})
    usage.setdefault('tool_calls_complete',True)
    uses_model=bool(config['model']) or config.get('hybrid',{}).get('weights',{}).get('lsa',0)>0
    default_tokens=None if uses_model else 0
    usage.setdefault('model_input_tokens',default_tokens);usage.setdefault('model_output_tokens',default_tokens)
    usage.setdefault('token_notes','Model usage unavailable; explain the reporting limitation.' if uses_model else 'No model used.')
    result['usage']=usage
    try:connection.send((result,tools.trace if tools else []))
    finally:connection.close()

def stop_process(p):
    if p.is_alive():
        try:
            if os.name=='posix' and os.getpgid(p.pid)==p.pid:os.killpg(p.pid,signal.SIGTERM)
            else:p.terminate()
        except ProcessLookupError:pass
        p.join(2)
    if p.is_alive():p.kill();p.join()

def load_resume(output,questions,info):
    """Reject incompatible resumes before changing any existing artifacts."""
    try:
        previous=json.loads((output/'run_info.json').read_text())
        defaults={'answer_policy':'v1',
                  'api_provider':'deepseek' if previous.get('model') else None}
        if previous.get('agent_module')=='agent.solution' and previous.get('model'):
            defaults.update(verification_policy='v1-numbered-clues',verification_thinking=False)
        if any(previous.get(key,defaults.get(key))!=value for key,value in info.items()):
            raise DataError('Resume configuration differs from the existing run')
        if read_jsonl(output/'questions.jsonl')!=questions:
            raise DataError('Resume questions differ from the existing run')
        results=read_jsonl(output/'results.jsonl')
    except (OSError,ValueError) as exc:
        raise DataError(f'Cannot resume incomplete run artifacts: {exc}') from exc
    known={q['id'] for q in questions};seen=set()
    for result in results:
        qid=result.get('id')
        if qid not in known or qid in seen:raise DataError('Invalid or duplicate resume result ID')
        seen.add(qid)
        if not (output/'traces'/f'{qid}.jsonl').is_file():raise DataError(f'Missing trace for {qid}')
    return results

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--split',choices=['train','dev','test'],default='dev')
    p.add_argument('--agent',default='examples.keyword_agent',help='Python module exposing solve(question, tools, config)')
    p.add_argument('--method-label',help='Method name/version; defaults to the adapter module name')
    p.add_argument('--method-type',choices=['retrieval','agent'],default='retrieval',help='Fixed retrieval pipeline (default) or model-directed tool interaction')
    p.add_argument('--model-label',help='Actual model identifier, if used; required for --method-type agent')
    p.add_argument('--api-provider',choices=['deepseek','siliconflow','dashscope'],default='deepseek',help='Answer API provider; SiliconFlow uses SILICONFLOW_API_KEY; DashScope uses IR2026_API_KEY and IR2026_API_BASE')
    p.add_argument('--retriever',choices=['bm25','hybrid'],default='bm25',help='Retrieval mode for agent.solution')
    p.add_argument('--hybrid-config',type=Path,help='Hybrid settings file; also accepts a preset containing a hybrid object')
    p.add_argument('--answer-policy',choices=['v1','verified'],default='v1',help='v1 is the selected baseline; verified enables experimental strict verification')
    p.add_argument('--retrieval-only',action='store_true',help='Evaluate retrieval with one source quote and no answer generation')
    p.add_argument('--resume',action='store_true',help='Reuse successful/no-answer results; retry failed and unfinished questions')
    p.add_argument('--limit',type=int)
    p.add_argument('--output',type=Path,default=ROOT/'output'/'demo')
    args=p.parse_args(argv)
    if args.limit is not None and args.limit<1:p.error('--limit must be positive')
    if args.method_label is not None and not args.method_label.strip():p.error('--method-label cannot be blank')
    if args.model_label is not None and not args.model_label.strip():p.error('--model-label cannot be blank')
    if args.model_label and args.model_label.startswith('local:'):
        p.error('Local model inference has been removed; use an API --model-label')
    if args.method_type=='agent' and not args.model_label:p.error('--method-type agent requires --model-label')
    if args.model_label=='offline-keyword-demo':p.error('Omit --model-label when no model is used')
    if (args.retriever!='bm25' or args.retrieval_only) and (args.agent!='agent.solution' or args.method_type!='retrieval'):
        p.error('--retriever hybrid and --retrieval-only require agent.solution with method-type retrieval')
    if args.retrieval_only and args.model_label:p.error('--retrieval-only does not call an answer model; omit --model-label')
    if args.hybrid_config and args.retriever!='hybrid':p.error('--hybrid-config requires --retriever hybrid')
    if args.answer_policy=='verified' and (args.agent!='agent.solution' or not args.model_label or args.retrieval_only):
        p.error('--answer-policy verified requires agent.solution with an answer model')
    if args.api_provider=='siliconflow' and not args.model_label:
        p.error('--api-provider siliconflow requires a remote --model-label')
    if args.agent=='agent.solution' or args.api_provider=='siliconflow':
        from agent.answer import load_api_environment
        load_api_environment(ROOT/'.env')
    if args.api_provider=='siliconflow' and not (os.environ.get('SILICONFLOW_API_KEY') or os.environ.get('IR2026_API_KEY')):
        p.error('Set SILICONFLOW_API_KEY in the environment or project .env before using SiliconFlow')
    config=load_policy(ROOT/'configs/default.json')
    config.update({'method_type':args.method_type,'model':args.model_label,
                   'retriever':args.retriever,'retrieval_only':args.retrieval_only,
                   'api_provider':args.api_provider,'answer_policy':args.answer_policy})
    retrieval_model=None
    if args.retriever=='hybrid':
        from agent.hybrid import INDEX_PATH as hybrid_path, load_config
        import numpy as np
        if not hybrid_path.is_file():p.error('Missing hybrid index; run python3 build_hybrid_index.py')
        config['hybrid']=load_config(args.hybrid_config)
        if config['hybrid']['weights']['lsa']>0:
            with np.load(hybrid_path,allow_pickle=False) as artifact:
                retrieval_model=json.loads(str(artifact['metadata']))['model']
        reranker=config['hybrid'].get('reranker')
        if reranker:
            retrieval_model='pairwise-ltr:' + reranker['model']
    questions=load_questions(ROOT/'datas'/f'{args.split}.jsonl')
    if args.limit is not None:questions=questions[:args.limit]
    info={'release':config['release'],'protocol_version':PROTOCOL_VERSION,
          'method':args.method_label or args.agent,'method_type':args.method_type,
          'agent_module':args.agent,'model':';'.join(x for x in [retrieval_model,args.model_label] if x) or None,
          'retriever':args.retriever,'retrieval_only':args.retrieval_only,
          'answer_policy':args.answer_policy,
          'api_provider':args.api_provider if args.model_label else None,
          'mode':'demo' if args.agent.startswith('examples.') else 'coursework',
          'split':args.split,'queries':len(questions),'limits':config['limits'],
          'python':sys.version.split()[0]}
    if args.retriever=='hybrid':info['hybrid']=config['hybrid']
    if args.agent=='agent.solution' and args.model_label:
        from agent.prompts import V1_VERIFICATION_POLICY
        policy=V1_VERIFICATION_POLICY
        if args.answer_policy=='verified':
            from agent.verification import POLICY
            policy=POLICY
        info.update(verification_policy=policy,
                    verification_thinking=(args.answer_policy=='verified' and args.api_provider=='siliconflow'
                                           and os.environ.get('IR2026_VERIFY_THINKING','0')=='1'))
    results=[]
    if args.resume:
        try:results=load_resume(args.output,questions,info)
        except DataError as exc:p.error(str(exc))
    else:
        if args.output.exists() and any(args.output.iterdir()):p.error('Use a new or empty output directory, or --resume')
        args.output.mkdir(parents=True,exist_ok=True);(args.output/'traces').mkdir()
        write_jsonl(args.output/'questions.jsonl',questions)
    info['execution_status']='running'
    (args.output/'run_info.json').write_text(json.dumps(info,ensure_ascii=False,indent=2)+'\n')
    ctx=mp.get_context('spawn');by_id={r['id']:r for r in results}
    for i,q in enumerate(questions,1):
        previous=by_id.get(q['id'])
        if previous and previous.get('status') in {'ok','no_answer'} and not previous.get('api_status_code'):
            print(f'{i}/{len(questions)} id={q["id"]} reused',flush=True)
            continue
        receiving,sending=ctx.Pipe(duplex=False)
        trace_path=args.output/'traces'/f'{q["id"]}.jsonl'
        proc=ctx.Process(target=worker,args=(sending,args.agent,q,ROOT/'corpus',config,trace_path))
        proc.start();sending.close()
        try:
            if receiving.poll(config['limits']['max_seconds']):
                try:result,trace=receiving.recv()
                except EOFError:result,trace=None,[]
            else:result,trace=None,[]
            if result is None:
                timed_out=proc.is_alive()
                stop_process(proc)
                if trace_path.exists():
                    try:trace=read_jsonl(trace_path)
                    except (DataError,OSError):trace=[]
                result={'id':q['id'],'answer':'','doc_ids':[],'evidence':[],
                        'status':'timeout' if timed_out else 'error',
                        'error':'Worker timed out or exited before returning a record.',
                        'usage':{'tool_calls':count_model_tool_calls(trace),'trace_events':len(trace),'tool_calls_complete':False,'elapsed_seconds':config['limits']['max_seconds'] if timed_out else 0,
                                 'model_input_tokens':None if args.model_label or retrieval_model else 0,'model_output_tokens':None if args.model_label or retrieval_model else 0,
                                 'token_notes':'Worker did not return model usage. Tool count covers persisted events and may miss an in-flight call.'}}
        finally:
            receiving.close();proc.join(1);stop_process(proc)
        by_id[result['id']]=result
        results=[by_id[item['id']] for item in questions if item['id'] in by_id]
        write_jsonl(args.output/'traces'/f'{q["id"]}.jsonl',trace)
        write_jsonl(args.output/'results.jsonl',results)
        print(f'{i}/{len(questions)} id={q["id"]} status={result["status"]}',flush=True)
        if result.get('api_status_code') in {401,402,403}:
            info.update(execution_status='api_error',stop_reason=result['error'])
            (args.output/'run_info.json').write_text(json.dumps(info,ensure_ascii=False,indent=2)+'\n')
            print(f'Stopped: {result["error"]}. Resolve the API account issue, then rerun with --resume.',file=sys.stderr)
            return 2
    info['execution_status']='complete'
    (args.output/'run_info.json').write_text(json.dumps(info,ensure_ascii=False,indent=2)+'\n')
    print(f'Results: {args.output / "results.jsonl"}')

if __name__=='__main__':sys.exit(main())
