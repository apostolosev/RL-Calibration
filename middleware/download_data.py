from finot_client import FINoTClient

import argparse



# Create an argument parser
def create_parser():
    parser = argparse.ArgumentParser(description="Script to upload the valve block calibration results to the middleware.")

    # Middleware connection arguments
    parser.add_argument("--host", type=str, default="api.optimai.finot.cloud")
    parser.add_argument("--username", type=str, default="optimai@f-in.eu")
    parser.add_argument("--password", type=str, default="optimaidemo")
    parser.add_argument("--entityID", type=str, default="valve-block-calibration")
    parser.add_argument("--dir_path", type=str, default="downloads")
    parser.add_argument("--option", type=str, default="dss")

    return parser

def main():
    parser = create_parser()
    args = parser.parse_args()
    fclient = FINoTClient(host=args.host, username=args.username, password=args.password)
    ngsi_entity = fclient.get_entity(args.entityID)
    if args.option == "dss":
        dss_results_path = ngsi_entity["dss_results"]["value"]
        fclient.download_file(args.entityID, dss_results_path, directory=args.dir_path)
    if args.option == "blockchain":
        blockchain_results_path = ngsi_entity["blockchain_results"]["value"]
        fclient.download_file(args.entityID, blockchain_results_path, directory=args.dir_path)


if __name__== "__main__":
    main()