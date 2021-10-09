__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

__version__ = '0.0.1'

import re

from jina.logging.logger import JinaLogger

import os
import sys
import click
import re
import asyncio
import torch


from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from subprocess import Popen, PIPE
from jina import Flow, Document, Executor, requests, DocumentArray
from helper import input_generator
from jina.logging.predefined import default_logger as logger

global latest_podcast_date_time
global quit

def config():
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    os.environ.setdefault('JINA_WORKSPACE', os.path.join(cur_dir, 'workspace'))
    os.environ.setdefault('JINA_WORKSPACE_MOUNT',
                          f'{os.environ.get("JINA_WORKSPACE")}:/workspace/workspace')
    os.environ.setdefault('JINA_LOG_LEVEL', 'INFO')
    if os.path.exists('lyrics-data/lyrics-data.csv'):
        os.environ.setdefault('JINA_DATA_FILE', 'lyrics-data/lyrics-data.csv')
    else:
        os.environ.setdefault('JINA_DATA_FILE', 'lyrics-data/lyrics-toy-data1000.csv')
    os.environ.setdefault('JINA_PORT', str(45678))
    os.environ.setdefault('JINA_PORT_2', str(40000))


# for index
def index(num_docs):

    flow = Flow.load_config('flows/index.yml')
    flow.plot("index_flow.jpg")
    with flow:
        input_docs = input_generator(num_docs=num_docs)
        data_path = os.path.join(os.path.dirname(__file__),
                                 os.environ.get('JINA_DATA_FILE', None))
        flow.logger.info(f'Indexing {data_path}')
        flow.post(on='/index', inputs=input_docs, request_size=10,
                  show_progress=True)


# for update index
def update_index(num_docs):

    current_podcast_date_time = 0

    os.system(f"echo 0 > index_updated.log")

    proc = Popen('find ./static/Images -type f -printf "%T+\t%p\n" | sort | tail -1', shell=True, stdout=PIPE)
    stdout = proc.stdout
    output = stdout.read()
    proc.terminate()
        
    os.system(f"echo output: {output[0:4]} {output[5:7]} {output[8:10]} {output[11:13]} {output[14:16]} {output[17:19]}")
    previous_upload_date_time = datetime(int(output[0:4]), int(output[5:7]), int(output[8:10]), int(output[11:13]), int(output[14:16]), int(output[17:19]))

    while True:

        proc = Popen('find ./static/Images -type f -printf "%T+\t%p\n" | sort | tail -1', shell=True, stdout=PIPE)
        stdout = proc.stdout
        output = stdout.read()
        proc.terminate()
        
        
        os.system(f"echo output: {output[0:4]} {output[5:7]} {output[8:10]} {output[11:13]} {output[14:16]} {output[17:19]}")
        current_upload_date_time = datetime(int(output[0:4]), int(output[5:7]), int(output[8:10]), int(output[11:13]), int(output[14:16]), int(output[17:19]))

        if(current_upload_date_time > previous_upload_date_time):
    
            os.system("python3 update_database.py") 
            previous_upload_date_time = current_upload_date_time

        proc =  Popen('find ./lyrics-data -type f -printf "%T+\t%p\n" | sort | tail -1', shell=True, stdout=PIPE)
        stdout = proc.stdout
        output = stdout.read()
        proc.terminate()
        
        os.system(f"echo output: {output[0:4]} {output[5:7]} {output[8:10]} {output[11:13]} {output[14:16]} {output[17:19]}")
        current_podcast_date_time = datetime(int(output[0:4]), int(output[5:7]), int(output[8:10]), int(output[11:13]), int(output[14:16]), int(output[17:19]))

        proc = Popen('cat latest_podcast_date_time.log', shell=True, stdout=PIPE)
        stdout = proc.stdout
        output = stdout.read()
        proc.terminate()

        latest_podcast_date_time = datetime(int(output[0:4]), int(output[5:7]), int(output[8:10]), int(output[11:13]), int(output[14:16]), int(output[17:19]))


        if(current_podcast_date_time > latest_podcast_date_time):

            os.system(f"echo Indexing has to be done again ")
            os.system(f"echo Query has to be restarted")

            proc = Popen('find ./lyrics-data -type f -printf "%T+\t%p\n" | sort | tail -1', shell=True, stdout=PIPE)
            stdout = proc.stdout
            output = stdout.read()
            proc.terminate()
    
            latest_podcast_date_time = datetime(int(output[0:4]), int(output[5:7]), int(output[8:10]), int(output[11:13]), int(output[14:16]), int(output[17:19]))
            os.system(f"echo {latest_podcast_date_time} > latest_podcast_date_time.log")

            flow = Flow.load_config('flows/update.yml')
            flow.plot("index_flow.jpg")
        
            with flow:
                input_docs = input_generator(num_docs=num_docs)
                data_path = os.path.join(os.path.dirname(__file__),
                                 os.environ.get('JINA_DATA_FILE', None))
                flow.logger.info(f'Indexing {data_path}')
                flow.post(on='/index', inputs=input_docs, request_size=10,
                          show_progress=True)

            os.system("echo 1 > index_updated.log")
            os.system("echo 1 > quit.log")

# for search
def query_question_answer():
    #flow = Flow.load_config('flows/query.yml').add(name='p1',needs='gateway')
    flow = Flow().load_config('flows/query_question_answer.yml')

    #Flow.load_config('flows/query.yml').plot("img.jpg")
    #flow.add(name='p1',needs='gateway') #mod:sharmaa4
    flow.plot("query_question_answer_flow.jpg")
    flow.rest_api = True
    flow.protocol = 'http'
    with flow:
        flow.block()
                


# for search
def query():
    #flow = Flow.load_config('flows/query.yml').add(name='p1',needs='gateway')
    flow = Flow().load_config('flows/query.yml')

    #Flow.load_config('flows/query.yml').plot("img.jpg")
    #flow.add(name='p1',needs='gateway') #mod:sharmaa4
    #flow.plot("query_flow.jpg")
    flow.rest_api = True
    flow.protocol = 'http'
    #with flow:
    #    flow.block()

    started = 0

    while True:

        try:
            if (started == 0):
                started = 1
                flow.start()
            else:
                os.system("echo Query running....")
        finally:

            proc = Popen('cat quit.log', shell=True, stdout=PIPE)
            stdout = proc.stdout
            output = stdout.read()
            proc.terminate()

            quit = int(output[0]) - 48
            os.system(f"echo {quit}")

            if(quit == 1):
                os.system(f"echo 0 > quit.log")
                flow.close()
                started = 0 


# for query update 
def query_update():
    #flow = Flow.load_config('flows/query.yml').add(name='p1',needs='gateway')
    flow = Flow().load_config('flows/query_update.yml')

    #Flow.load_config('flows/query.yml').plot("img.jpg")
    #flow.add(name='p1',needs='gateway') #mod:sharmaa4
    flow.plot("query_flow.jpg")
    flow.rest_api = True
    flow.protocol = 'http'
    with flow :
        flow.post('/search', inputs="test", parameters={'lookup_type': 'parent'}, return_results=True)


def query_text():
    def print_result(response):
        doc = response.docs[0]
        for index, parent in enumerate(doc.matches):
            print(f'Parent {index}: Song Name: {parent.tags["SName"]}\n{parent.text}')
        for index, chunk in enumerate(doc.chunks):
            print(f'Chunk {index}: {chunk.text}')
            for match in chunk.matches:
                print(f'\tMatch: {match.text}')

    os.system(f"echo output")
    f = Flow().load_config('flows/query.yml')
    with f:
        search_text = input('Please type a sentence: ')
        #search_text = "Hello"
        doc = Document(content=search_text, mime_type='text/plain')
        response = f.post('/search', inputs=doc, parameters={'lookup_type': 'parent'}, return_results=True)
        print_result(response[0].data)

class _LMDBHandler:
    def __init__(self, file, map_size):
        # see https://lmdb.readthedocs.io/en/release/#environment-class for usage
        self.file = file
        self.map_size = map_size

    @property
    def env(self):
        return self._env

    def __enter__(self):
        self._env = lmdb.Environment(
            self.file,
            map_size=self.map_size,
            subdir=False,
            readonly=False,
            metasync=True,
            sync=True,
            map_async=False,
            mode=493,
            create=True,
            readahead=True,
            writemap=False,
            meminit=True,
            max_readers=126,
            max_dbs=0,  # means only one db
            max_spare_txns=1,
            lock=True,
        )
        return self._env

    def __exit__(self, exc_type, exc_val, exc_tb):
        if hasattr(self, '_env'):
            self._env.close()


class LMDBStorage(Executor):
    """An lmdb-based Storage Indexer for Jina

    For more information on lmdb check their documentation: https://lmdb.readthedocs.io/en/release/
    """

    def __init__(
        self,
        map_size: int = 1048576000,  # in bytes, 1000 MB
        default_traversal_paths: List[str] = ['r'],
        dump_path: str = None,
        default_return_embeddings: bool = True,
        *args,
        **kwargs,
    ):
        """
        :param map_size: the maximal size of teh database. Check more information at
            https://lmdb.readthedocs.io/en/release/#environment-class
        :param default_traversal_paths: fallback traversal path in case there is not traversal path sent in the request
        :param default_return_embeddings: whether to return embeddings on search or not
        """
        super().__init__(*args, **kwargs)
        self.map_size = map_size
        self.default_traversal_paths = default_traversal_paths
        self.file = os.path.join(self.workspace, 'db.lmdb')
        if not os.path.exists(self.workspace):
            os.makedirs(self.workspace)
        self.logger = get_logger(self)

        self.dump_path = dump_path or kwargs.get('runtime_args', {}).get(
            'dump_path', None
        )
        if self.dump_path is not None:
            self.logger.info(f'Importing data from {self.dump_path}')
            ids, metas = import_metas(self.dump_path, str(self.runtime_args.pea_id))
            da = DocumentArray()
            for id, meta in zip(ids, metas):
                serialized_doc = Document(meta)
                serialized_doc.id = id
                da.append(serialized_doc)
            self.index(da, parameters={'traversal_paths': ['r']})
        self.default_return_embeddings = default_return_embeddings

    def _handler(self):
        # required to create a new connection to the same file
        # on each subprocess
        # https://github.com/jnwatson/py-lmdb/issues/289
        return _LMDBHandler(self.file, self.map_size)

    @requests(on='/index')
    def index(self, docs: DocumentArray, parameters: Dict, **kwargs):
        """Add entries to the index

        :param docs: the documents to add
        :param parameters: parameters to the request
        """
        traversal_paths = parameters.get(
            'traversal_paths', self.default_traversal_paths
        )
        if docs is None:
            return
        with self._handler() as env:
            with env.begin(write=True) as transaction:
                for d in docs.traverse_flat(traversal_paths):
                    transaction.put(d.id.encode(), d.SerializeToString())

    @requests(on='/update')
    def update(self, docs: DocumentArray, parameters: Dict, **kwargs):
        """Update entries from the index by id

        :param docs: the documents to update
        :param parameters: parameters to the request
        """
        traversal_paths = parameters.get(
            'traversal_paths', self.default_traversal_paths
        )
        if docs is None:
            return
        with self._handler() as env:
            with env.begin(write=True) as transaction:
                for d in docs.traverse_flat(traversal_paths):
                    # TODO figure out if there is a better way to do update in LMDB
                    # issue: the defacto update method is an upsert (if a value didn't exist, it is created)
                    # see https://lmdb.readthedocs.io/en/release/#lmdb.Cursor.replace
                    if transaction.delete(d.id.encode()):
                        transaction.replace(d.id.encode(), d.SerializeToString())

    @requests(on='/delete')
    def delete(self, docs: DocumentArray, parameters: Dict, **kwargs):
        """Delete entries from the index by id

        :param docs: the documents to delete
        :param parameters: parameters to the request
        """
        traversal_paths = parameters.get(
            'traversal_paths', self.default_traversal_paths
        )
        if docs is None:
            return
        with self._handler() as env:
            with env.begin(write=True) as transaction:
                for d in docs.traverse_flat(traversal_paths):
                    transaction.delete(d.id.encode())

    @requests(on='/search')
    def search(self, docs: DocumentArray, parameters: Dict, **kwargs):
        """Retrieve Document contents by ids

        :param docs: the list of Documents (they only need to contain the ids)
        :param parameters: the parameters for this request
        """
        traversal_paths = parameters.get(
            'traversal_paths', self.default_traversal_paths
        )
        return_embeddings = parameters.get(
            'return_embeddings', self.default_return_embeddings
        )
        if docs is None:
            return
        docs_to_get = docs.traverse_flat(traversal_paths)
        with self._handler() as env:
            with env.begin(write=True) as transaction:
                for i, d in enumerate(docs_to_get):
                    id = d.id
                    serialized_doc = Document(transaction.get(d.id.encode()))
                    if not return_embeddings:
                        serialized_doc.pop('embedding')
                    d.update(serialized_doc)
                    d.id = id
                    print(d.id.data)
                    

    @requests(on='/dump')
    def dump(self, parameters: Dict, **kwargs):
        """Dump data from the index

        Requires
        - dump_path
        - shards
        to be part of `parameters`

        :param parameters: parameters to the request"""
        path = parameters.get('dump_path', None)
        if path is None:
            self.logger.error('parameters["dump_path"] was None')
            return

        shards = parameters.get('shards', None)
        if shards is None:
            self.logger.error('parameters["shards"] was None')
            return
        shards = int(shards)

        export_dump_streaming(path, shards, self.size, self._dump_generator())

    @property
    def size(self):
        """Compute size (nr of elements in lmdb)"""
        with self._handler() as env:
            with env.begin(write=True) as transaction:
                stats = transaction.stat()
                return stats['entries']

    def _dump_generator(self):
        with self._handler() as env:
            with env.begin(write=True) as transaction:
                cursor = transaction.cursor()
                cursor.iternext()
                iterator = cursor.iternext(keys=True, values=True)
                for it in iterator:
                    id, data = it
                    doc = Document(data)
                    yield id.decode(), doc.embedding, LMDBStorage._doc_without_embedding(
                        doc
                    ).SerializeToString()

    @staticmethod
    def _doc_without_embedding(d):
        new_doc = Document(d, copy=True)
        new_doc.ClearField('embedding')
        return new_doc





class Answer_Machine(Executor):
    """
    :class:`Sentencizer` split the text on the doc-level
    into sentences on the chunk-level with a rule-base strategy.
    The text is split by the punctuation characters listed in ``punct_chars``.
    The sentences that are shorter than the ``min_sent_len``
    or longer than the ``max_sent_len`` after stripping will be discarded.
    """

    def __init__(
        self,
        min_sent_len: int = 1,
        max_sent_len: int = 512,
        punct_chars: Optional[List[str]] = None,
        uniform_weight: bool = True,
        traversal_paths: Tuple[str] = ('r',),
        *args,
        **kwargs
    ):
        """
        :param min_sent_len: the minimal number of characters,
            (including white spaces) of the sentence, by default 1.
        :param max_sent_len: the maximal number of characters,
            (including white spaces) of the sentence, by default 512.
        :param punct_chars: the punctuation characters to split on,
            whatever is in the list will be used,
            for example ['!', '.', '?'] will use '!', '.' and '?'
        :param uniform_weight: the definition of it should have
            uniform weight or should be calculated
        :param traversal_paths: traverse path on docs, e.g. ['r'], ['c']
        """
        super().__init__(*args, **kwargs)
        self.min_sent_len = min_sent_len
        self.max_sent_len = max_sent_len
        self.punct_chars = punct_chars
        self.uniform_weight = uniform_weight
        self.logger = JinaLogger(self.__class__.__name__)
        self.traversal_paths = traversal_paths
        if not punct_chars:
            self.punct_chars = [
                '!',
                '.',
                '?',
                '։',
                '؟',
                '۔',
                '܀',
                '܁',
                '܂',
                '‼',
                '‽',
                '⁇',
                '⁈',
                '⁉',
                '⸮',
                '﹖',
                '﹗',
                '！',
                '．',
                '？',
                '｡',
                '。',
                '\n',
            ]
        if self.min_sent_len > self.max_sent_len:
            self.logger.warning(
                'the min_sent_len (={}) should be smaller or equal to the max_sent_len (={})'.format(
                    self.min_sent_len, self.max_sent_len
                )
            )
        self._slit_pat = re.compile(
            r'\s*([^{0}]+)(?<!\s)[{0}]*'.format(''.join(set(self.punct_chars)))
        )

    @requests
    def segment(self, docs: Optional[DocumentArray], parameters: Dict, **kwargs):
        """
        Split the text into sentences.
        :param docs: Documents that contain the text
        :param parameters: Dictionary of parameters
        :param kwargs: Additional keyword arguments
        :return: a list of chunk dicts with the split sentences
        """
        if not docs:
            return
        traversal_path = parameters.get('traversal_paths', self.traversal_paths)
        flat_docs = docs.traverse_flat(traversal_path)
        for doc in flat_docs:
            text = doc.text
            print(text.split("\n")[0])

            tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
            model = AutoModelForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")

            text = doc.text.split("\n")[1][0:1500]
            print(type(doc.text.split("\n")[1]))
            print(len(doc.text.split("\n")[1]))

            #text = [str(doc.text.split("\n")[1])]

            #questions = [
            #"How many pretrained models are available in Transformers?",
            #"What does Transformers provide?",
            #"Transformers provides interoperability between which frameworks?",
            #]

            questions = [doc.text.split("\n")[0]]
            for question in questions:
                inputs = tokenizer(question,text , add_special_tokens=True, return_tensors="pt")
                input_ids = inputs["input_ids"].tolist()[0]

                outputs = model(**inputs)
                answer_start_scores = outputs.start_logits
                answer_end_scores = outputs.end_logits

                answer_start = torch.argmax(
                answer_start_scores
            )  # Get the most likely beginning of answer with the argmax of the score
            answer_end = torch.argmax(answer_end_scores) + 1  # Get the most likely end of answer with the argmax of the score

            answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))

            print(f"Question: {question}")
            print(f"Answer: {answer}")

            """
            ret = [
                (m.group(0), m.start(), m.end())
                for m in re.finditer(self._slit_pat, text)
            ]
            if not ret:
                ret = [(text, 0, len(text))]
            for ci, (r, s, e) in enumerate(ret):
                f = re.sub('\n+', ' ', r).strip()
                f = f[: self.max_sent_len]
                if len(f) > self.min_sent_len:
                    doc.chunks.append(
                        Document(
                            text=f,
                            offset=ci,
                            weight=1.0 if self.uniform_weight else len(f) / len(text),
                            location=[s, e],
                        )
                    )

            """


class MyExecutor(Executor):

    @requests(on='/search')
    def print_dummy(self,**kwargs):

         
        
        stdout = Popen('cat index_updated.log', shell=True, stdout=PIPE).stdout
        output = stdout.read()

        index_updated = int(output[0])
        os.system(f"echo {index_updated}")


        """ 
        if(index_updated == 1):

            os.system(f"echo Indexing updated ")
            stdout = Popen('python3 app.py -t query_text', shell=True, stdout=PIPE)
            output = stdout.read()

            os.system(f"echo 0 > index_updated.log")
        """

        """
            flow = Flow().load_config('flows/update.yml')
            #flow.plot("index_flow.jpg")
            with flow:
                input_docs = input_generator(num_docs=10000)
                data_path = os.path.join(os.path.dirname(__file__),os.environ.get('JINA_DATA_FILE', None))
                flow.logger.info(f'Indexing {data_path}')
                flow.post(on='/update', inputs=input_docs, request_size=10,show_progress=True)

        """
        #latest_podcast = os.system('find ./podcast_dataset/Podcast -type f -printf "%T+\t%p\n" | sort | tail -1')
        #latest_podcast = str(latest_podcast)
        #latest_podcast_date = latest_podcast[0:10]
        #latest_podcast_time = latest_podcast[12:20]
        #os.system(f"echo {latest_podcast}") 
        #os.system("pwd")


@click.command()
@click.option('--task', '-t',
              type=click.Choice(['index', 'query', 'query_text', 'update_index', 'query_update', 'query_question_answer'], case_sensitive=False))
@click.option('--num_docs', '-n', default=10000)
def main(task, num_docs):
    config()
   
    latest_podcast_date_time = 0
    
    stdout = Popen('find ./lyrics-data -type f -printf "%T+\t%p\n" | sort | tail -1', shell=True, stdout=PIPE).stdout
    output = stdout.read()
    
    latest_podcast_date_time = datetime(int(output[0:4]), int(output[5:7]), int(output[8:10]), int(output[11:13]), int(output[14:16]), int(output[17:19]))
    os.system(f"echo {latest_podcast_date_time} > latest_podcast_date_time.log")
    

    workspace = os.environ["JINA_WORKSPACE"]
    #flow = Flow.load_config('flows/index.yml').plot("index_flow.jpg")

    if task == 'index':
        """
        if os.path.exists(workspace):
            logger.error(f'\n +---------------------------------------------------------------------------------+ \
                    \n |                                   🤖🤖🤖                                        | \
                    \n | The directory {workspace} already exists. Please remove it before indexing again. | \
                    \n |                                   🤖🤖🤖                                        | \
                    \n +---------------------------------------------------------------------------------+')
            sys.exit(1)
        """    
        index(num_docs)
    elif task == 'query_question_answer':
        query_question_answer()
        
    elif task == 'query':
        query()
        
    elif task == 'query_text':
        query_text()

    elif task == 'query_update':
        query_update()

    elif task == 'update_index':
        update_index(num_docs)

    else:
        raise NotImplementedError(
            f'Unknown task: {task}.')


if __name__ == '__main__':
    main()
