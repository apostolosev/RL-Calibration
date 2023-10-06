from finot_client import FINoTClient

import os
import argparse

# Create an argument parser
def create_parser():
    parser = argparse.ArgumentParser(description="Script to upload the valve block calibration results to the middleware.")

    # Middleware connection arguments
    parser.add_argument("--host", type=str, default="api.optimai.finot.cloud")
    parser.add_argument("--username", type=str, default="optimai@f-in.eu")
    parser.add_argument("--password", type=str, default="optimaidemo")
    parser.add_argument("--entityID", type=str, default="valve-block-calibration")
    parser.add_argument("--serial_number", type=int, default=0)

    # DSS results
    parser.add_argument("--dss_results", type=str, default="1675950261486.json")
    parser.add_argument("--dss_root", type=str, default="middleware/data/dss")

    # Blockchain results
    parser.add_argument("--blockchain_results", type=str, default="1675950261486.json")
    parser.add_argument("--blockchain_root", type=str, default="middleware/data/blockchain")

    parser.add_argument("--action", type=str, default="c")

    return parser


class MiddlewareConnection:
    def __init__(self,
                 host,
                 username,
                 password,
                 entityID, 
                 serial_number):
        self.host = host
        self.username = username
        self.password = password
        self.entityID = entityID
        self.serial_number = serial_number
        self.fclient = FINoTClient(host=host, username=username, password=password)
        self.json_body = {
            "id": self.entityID,
            "type": "CERTH_RLCalibration",
            "static_attributes": [
                {"name": "serial_number", "type": "Text", "value": self.serial_number}
            ],
            "attributes": [
                {"name": "dss_results", "type": "File", "value": "dss_results"},
                {"name": "blockchain_results", "type": "File", "value": "blockchain_results"}
            ]
        }

    def create_entity(self, dss_results, blockchain_results, dss_root, blockchain_root):
        self.fclient.create_entity_legacy(self.json_body,
                                          {"dss_results": (dss_results, open(os.path.join(dss_root, dss_results), "rb"), "text/json"), 
                                          "blockchain_results": (blockchain_results, open(os.path.join(blockchain_root, blockchain_results), "rb"), "text/json")})

    def update_entity(self, dss_results, blockchain_resuts, dss_root, blockchain_root):
        json_body = {"dss_results": {"type": "File", "value": "dss_results"}}
        self.fclient.update_entity(self.entityID, json_body,
                                   {"dss_results": (dss_results, open(os.path.join(dss_root, dss_results), "rb"), "text/json")})
        json_body = {"blockchain_results" : {"type": "File", "value": "blockchain_results"}}
        self.fclient.update_entity(self.entityID, json_body, 
                                   {"blockchain_results": (blockchain_resuts, open(os.path.join(blockchain_root, blockchain_resuts), "rb"), "text/json")})

    def get_entity(self):
        ngsi_entity = self.fclient.get_entity(self.entityID)
        return ngsi_entity

    def get_entity_historical(self):
        ngsi_entities = self.fclient.get_entity_historical(self.entityID, attrs=["dss_results", "blockchain_results"], lastN=2)
        return ngsi_entities

    def download_files(self):
        ngsi_entity = self.fclient.get_entity(self.entityID)
        dss_results_path = ngsi_entity["dss_results"]["value"]
        blockchain_results_path = ngsi_entity["blockchain_results"]["value"]
        self.fclient.download_file(self.entityID, dss_results_path, directory="./downloads/dss")
        self.fclient.download_file(self.entityID, blockchain_results_path, directory="./downloads/blockchain/")

    def delete_entity(self):
        self.fclient.delete_entity(self.entityID)


def main():
    parser = create_parser()
    args = parser.parse_args()
    middlewareConnection = MiddlewareConnection(host=args.host,
                                  username=args.username,
                                  password=args.password,
                                  entityID=args.entityID,
                                  serial_number=args.serial_number)

    if args.action == "create":
        middlewareConnection.create_entity(args.dss_results, 
                                      args.blockchain_results, 
                                      args.dss_root, 
                                      args.blockchain_root)
    elif args.action == "update":
        middlewareConnection.update_entity(args.dss_results, 
                                      args.blockchain_results, 
                                      args.dss_root, 
                                      args.blockchain_root)
    elif args.action == "download":
        middlewareConnection.download_files()
    elif args.action == "delete":
        middlewareConnection.delete_entity()
    else:
        print("Invalid action selected")


if __name__ == "__main__":
    main()
