from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import pandas as pd
import joblib
import os

# Create your views here.

# Load the trained pipeline
pipeline_path = os.path.join(os.path.dirname(__file__), 'logistic_regression_tuned_pipeline.pkl')
pipeline = joblib.load(pipeline_path)

class PredictApproval(APIView):
    def post(self, request, *args, **kwargs):
        # Parse the input JSON data
        data = request.data

        # Convert input data into a Pandas DataFrame
        try:
            input_data = pd.DataFrame(data)
        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)

        # Validate required columns
        required_columns = ['Num_Children', 'Gender', 'Income', 'Own_Car', 'Own_Housing']
        missing_columns = [col for col in required_columns if col not in input_data.columns]
        if missing_columns:
            return Response({'error': f'Missing required columns: {missing_columns}'}, status=status.HTTP_400_BAD_REQUEST)

        # Perform prediction
        try:
            predictions = pipeline.predict(input_data)
        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        # Return predictions
        return Response({'predictions': predictions.tolist()}, status=status.HTTP_200_OK)
