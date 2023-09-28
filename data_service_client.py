import grpc

import data_service_pb2
import data_service_pb2_grpc


def run():
    # Create a gRPC channel and stub
    channel = grpc.insecure_channel('localhost:50051')
    stub = data_service_pb2_grpc.DataServiceStub(channel)

    # Call the gRPC methods
    response1 = stub.GetDatasetData(data_service_pb2.GetDatasetDataRequest())
    print("Dataset Data:")
    print(response1.data)

    dataset_id = 'eugene'
    response2 = stub.CheckWashes(data_service_pb2.WashesRequest(dataset_id=dataset_id))
    print(f"Rains Extraction Result for dataset  '{dataset_id}':")

    print(response2.washes)

    dataset_id = 'eugene'
    start_date = '2012-12-20'
    end_date = '2014-01-20'
    request3 = data_service_pb2.CPDetectionRequest(
        dataset_id=dataset_id,
        start_date=start_date,
        end_date=end_date,
        w_train=30,
        wa1=10,
        wa2=5,
        wa3=10,
        thrsh=1,
        cp_starts=[],
        cp_ends=[]
    )
    response3 = stub.CPDetection(request3)
#     print(f"Changepoint Detection Result for dataset '{dataset_id}':")

    print(response3.result)

    dataset_id = 'eugene'
    start_date = '2012-12-20'
    end_date = '2014-01-20'
    request4 = data_service_pb2.ExtractRainsRequest(
        dataset_id=dataset_id,
        start_date=start_date,
        end_date=end_date
    )
    response4 = stub.ExtractRains(request4)
    print("Washes for dataset '" + dataset_id + "':")
    print(response4.result)

    dataset_id = 'eugene'
    start_date = '2012-12-20'
    end_date = '2014-01-20'
    request5 = data_service_pb2.CalculatePowerIndexRequest(
        dataset_id=dataset_id,
        start_date=start_date,
        end_date=end_date,
        weeks_train=4,
        cp_starts=[],
        cp_ends=[],
        query_modelar=False
    )
    response5 = stub.CalculatePowerIndex(request5)
    print(f"Power Index Calculation Result for dataset '{dataset_id}':")
    print(f"Output file: {response5.filename}")
    
    ##forecasting
    dataset_id = 'beico'
    start_date = "2021-07-01"
    end_date= "2022-08-01"
    
    request6 = data_service_pb2.ForecastingRequest(
        dataset_id=dataset_id,
        start_date=start_date,
        end_date=end_date,
        lags=3,
        future_steps=48,
        query_modelar=False
    )
    response6 = stub.Forecasting(request6)
    print(f"Forecasting Calculation Result for dataset '{dataset_id}':")
#     print(f"Output file: {response6.filename}")

    dataset_id = 'bbz2'
    start_date = None
    end_date=None
    
    request7 = data_service_pb2.EstimateYawMisalignmentRequest(
        dataset_id=dataset_id,
        start_date=start_date,
        end_date=end_date,
        window=2,
        query_modelar=False
    )
    response7 = stub.EstimateYawMisalignment(request7)
#     print(f"Yaw Misalignment Estimation Result for dataset  '{dataset_id}':")
#     print(f"Output file: {response6.filename}")


if __name__ == '__main__':
    run()
