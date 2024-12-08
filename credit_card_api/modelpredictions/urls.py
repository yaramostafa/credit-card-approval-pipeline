from django.urls import path
from .views import PredictApproval

urlpatterns = [
    path('predict/', PredictApproval.as_view(), name='predict-approval'),
]
