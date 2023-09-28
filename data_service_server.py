from concurrent import futures
import grpc
import json
import time
import data_service_pb2
import data_service_pb2_grpc

from changepoint_detection import run_cp_detection, read_wash_csv
from rains import extract_rains
from power_index import calculate_pi
from yaw_misalignment import estimate_yaw
from forecasting_estimation import *

_ONE_DAY_IN_SECONDS = 60 * 60 * 24


class DataServiceServicer(data_service_pb2_grpc.DataServiceServicer):
    def __init__(self):
        self.data_dict = {
            "eugene": {"id": 1, "name": "Eugene", "washes": True},
            "cocoa": {"id": 2, "name": "Cocoa", "washes": False},
        }

        self.path_dict = {
            "eugene": {"data": "/opt/more-workspace/solar/eugene.csv", "washes": "/opt/more-workspace/solar/eugene_washes.csv"},
            "cocoa": {"data": "/opt/more-workspace/solar/data/cocoa.csv", "washes": ""},
        }

        self.path_dict_yaw = {"bbz1big": {"data": "/opt/more-workspace/BEBEZE01_scada_high_frequency.parquet"}}
        
        ###New for forecasting
        self.path_dict_forecasting={"beico":{"data":"/home/pgidarakos/Forecasting_30min/data1h.csv"}}

    def GetDatasetData(self, request, context):
        data = json.dumps(self.data_dict)
        return data_service_pb2.DatasetDataResponse(data=data)

    def CheckWashes(self, request, context):
        dataset_id = request.dataset_id
        if dataset_id in self.data_dict:
            res = read_wash_csv(self.path_dict[dataset_id]["washes"])
            res_json = res.astype(str).to_json()
            return data_service_pb2.WashesResponse(washes=res_json)
        return data_service_pb2.WashesResponse()

    def CPDetection(self, request, context):
        dataset_id = request.dataset_id
        start_date = request.start_date
        end_date = request.end_date
        w_train = request.w_train if request.w_train else 30
        wa1 = request.wa1 if request.wa1 else 10
        wa2 = request.wa2 if request.wa2 else 5
        wa3 = request.wa3 if request.wa3 else 10
        thrsh = request.thrsh if request.thrsh else 1
        custom_cp_starts = request.cp_starts or []
        custom_cp_ends = request.cp_ends or []
        path = self.path_dict[dataset_id]["data"]
        wash_path = self.path_dict[dataset_id]["washes"]
        result_df = run_cp_detection(
            path=path,
            wash_path=wash_path,
            start_date=start_date,
            end_date=end_date,
            w_train=w_train,
            wa1=wa1,
            wa2=wa2,
            wa3=wa3,
            thrsh=thrsh,
            custom_cp_starts=custom_cp_starts,
            custom_cp_ends=custom_cp_ends,
        ).sort_values(by="Score")
        result_json = result_df.astype(str).to_json()
        return data_service_pb2.CPDetectionResponse(result=result_json)

    def ExtractRains(self, request, context):
        dataset_id = request.dataset_id
        path = self.path_dict[dataset_id]["data"]
        start_date = request.start_date
        end_date = request.end_date
        result_df = extract_rains(path, start_date, end_date)
        result_json = result_df.astype(str).to_json()
        return data_service_pb2.ExtractRainsResponse(result=result_json)

    def CalculatePowerIndex(self, request, context):
        dataset_id = request.dataset_id
        start_date = request.start_date
        end_date = request.end_date
        weeks_train = request.weeks_train if request.weeks_train else 4
        cp_starts = request.cp_starts or []
        cp_ends = request.cp_ends or []
        query_modelar = request.query_modelar if request.query_modelar else False
        path = self.path_dict[dataset_id]["data"]
        result_df = calculate_pi(
            path=path,
            start_date=start_date,
            end_date=end_date,
            weeks_train=weeks_train,
            cp_starts=cp_starts,
            cp_ends=cp_ends,
            query_modelar=query_modelar,
            dataset_id=dataset_id,
        )
        result_json = result_df.astype(str).to_json()
        return data_service_pb2.CalculatePowerIndexResponse(result=result_json)
        #filename = f"{dataset_id}_power_index.csv"
        #return data_service_pb2.CalculatePowerIndexResponse(filename=filename)

    def Forecasting(self, request, context):
        dataset_id = request.dataset_id
        start_date = request.start_date
        end_date = request.end_date
        lags = request.lags if request.lags else 3
        future_steps = request.future_steps if request.future_steps else 48
        query_modelar = request.query_modelar if request.query_modelar else False
        path = self.path_dict_forecasting[dataset_id]["data"]
        result_df = estimate(
            path=path,
            start_date=start_date,
            end_date=end_date,
            lags=lags,
            future_steps=future_steps,
            query_modelar=query_modelar,
            dataset_id=dataset_id,
        )
#         filename = f"{dataset_id}_forecasting_estimation.csv"
#         path_out = f"./outputs/{filename}"
#         result_df.to_csv(path_out)
        return data_service_pb2.ForecastingResponse(filename='nothing_yet')

    def EstimateYawMisalignment(self, request, context):
        dataset_id = request.dataset_id
        start_date = request.start_date
        end_date = request.end_date
        window = request.window if request.window else 2
        query_modelar = request.query_modelar if request.query_modelar else False
        path = self.path_dict_yaw[dataset_id]["data"]
        result_df = estimate_yaw(
            path=path,
            start_date=start_date,
            end_date=end_date,
            window=window,
            query_modelar=query_modelar,
            dataset_id=dataset_id,
        )
        filename = f"{dataset_id}_yaw_estimation.csv"
        path_out = f"./outputs/{filename}"
        result_json = result_df.astype(str).to_json()
        return data_service_pb2.EstimateYawMisalignmentResponse(result=result_json)


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    data_service_pb2_grpc.add_DataServiceServicer_to_server(DataServiceServicer(), server)
    server.add_insecure_port("[::]:50051")
    server.start()
    print("gRPC server started on port 50051...")
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == "__main__":
    serve()
