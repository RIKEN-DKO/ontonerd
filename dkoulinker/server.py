import json
from http.server import BaseHTTPRequestHandler

from dkoulinker.entity_linker import EntityLinker
from flair.models import SequenceTagger
from dkoulinker.entity_ranking import DictionaryRanking, QueryEntityRanking


API_DOC = "API_DOC"

"""
Class/function combination that is used to setup an API that can be used for e.g. GERBIL evaluation.
"""


def make_handler(e_linker):
    class GetHandler(BaseHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            self.e_linker=e_linker

            

            super().__init__(*args, **kwargs)

        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(
                bytes(
                    json.dumps(
                        {
                            "schemaVersion": 1,
                            "label": "status",
                            "message": "up",
                            "color": "green",
                        }
                    ),
                    "utf-8",
                )
            )
            return

        def do_HEAD(self):
            # send bad request response code
            self.send_response(400)
            self.end_headers()
            self.wfile.write(bytes(json.dumps([]), "utf-8"))
            return

        def do_POST(self):
            """
            Returns response.

            :return:
            """
            try:
                content_length = int(self.headers["Content-Length"])
                post_data = self.rfile.read(content_length)
                self.send_response(200)
                self.end_headers()

                text= self.read_json(post_data)
                response = self.generate_response(text)

                self.wfile.write(bytes(json.dumps(response), "utf-8"))
            except Exception as e:
                print(f"Encountered exception: {repr(e)}")
                self.send_response(400)
                self.end_headers()
                self.wfile.write(bytes(json.dumps([]), "utf-8"))
            return

        def read_json(self, post_data):
            """
            Reads input JSON message.

            :return: document text and spans.
            """

            data = json.loads(post_data.decode("utf-8"))
            text = data["text"]
            text = text.replace("&amp;", "&")
            return text

        def generate_response(self, text):
            """
            Generates response for API. Can be either ED only or EL, meaning end-to-end.

            :return: list of tuples for each entity found.
            """

            if len(text) == 0:
                return []


            # Process result.
            result = self.e_linker.link_entities(text)

            # Singular document.
            if len(result) > 0:
                # return [*result.values()][0]
                return result

            return []

    return GetHandler


if __name__ == "__main__":
    import argparse
    import pickle
    from http.server import HTTPServer
    import config

    p = argparse.ArgumentParser()

    p.add_argument("--pem_file", default=config.ONTO_PEM)
    p.add_argument("--entity2description_file",
                   default=config.ONTO_ENTITY2DESCRIPTION)
    p.add_argument("--mentionfreq_file",
                   default=config.ONTO_MENTION_FREQ)
    p.add_argument("--tagger_file",
                   default=config.ONTO_TAGGER)

    p.add_argument("--bind", "-b", metavar="ADDRESS", default="0.0.0.0")
    p.add_argument("--port", "-p", default=5555, type=int)
    args = p.parse_args()

    ##### loading EL
    #loading dicitonary of commonness,
    print('Loading mention2pem dictionary ...')
    handle = open(args.pem_file, 'rb')
    mention2pem = pickle.load(handle)

    print('Loading entity description dictionary ...')
    handle_desc = open(args.entity2description_file, 'rb')
    entity2description = pickle.load(handle_desc)
    print('NUmber of entities: ', len(entity2description))

    print('Loading dictionary of term frequency ...')
    handle_desc = open(args.mentionfreq_file, 'rb')
    mention2freq = pickle.load(handle_desc)
    print('Number of term in the collection: ', len(mention2freq))

    #given by create_term_req
    collection_size_terms = len(mention2pem)
    tagger = SequenceTagger.load(args.tagger_file)

    dictionarysearch_strategy = DictionaryRanking(mention2pem)
    queryranking_strategy = QueryEntityRanking(
        entity2description=entity2description,
        mention_freq=mention2freq,
        mention2pem=mention2pem
    )
    e_linker = EntityLinker(
        ranking_strategy=queryranking_strategy,
        ner_model=tagger,
        mention2pem=mention2pem,
        prune_overlapping_method='large_text'
    )
    ################

    server_address = (args.bind, args.port)
    server = HTTPServer(
        server_address,
        make_handler(e_linker),
    )

    try:
        print("Ready for listening.")
        server.serve_forever()
    except KeyboardInterrupt:
        exit(0)
