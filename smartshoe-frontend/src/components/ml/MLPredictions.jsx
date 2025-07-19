import React, { useState } from 'react'
import { useQuery, useMutation } from '@tanstack/react-query'
import { mlApiService } from '../../services/mlApi'
import { useAuth } from '../../contexts/AuthContext'
import {
  Brain,
  Activity,
  AlertTriangle,
  TrendingUp,
  Zap,
  RefreshCw,
  CheckCircle,
  AlertCircle,
  Info,
  BarChart3
} from 'lucide-react'
import LoadingSpinner from '../common/LoadingSpinner'
import toast from 'react-hot-toast'

const MLPredictions = ({ patientId, readingId = null }) => {
  const { canAccess } = useAuth()
  const [selectedModel, setSelectedModel] = useState('all')
  const [isRunningAnalysis, setIsRunningAnalysis] = useState(false)

  // Fetch existing predictions
  const { data: predictions, isLoading, refetch } = useQuery({
    queryKey: ['ml-predictions', patientId, selectedModel],
    queryFn: () => mlApiService.predictions.getPatientPredictions(patientId, selectedModel === 'all' ? null : selectedModel),
    enabled: canAccess('PROVIDER') && !!patientId,
    select: data => data?.data?.predictions || []
  })

  // Mutation for running new predictions
  const neuropathyMutation = useMutation({
    mutationFn: () => mlApiService.predictions.predictNeuropathyProgression(patientId, readingId),
    onSuccess: (data) => {
      toast.success('Neuropathy progression analysis completed')
      refetch()
    },
    onError: (error) => {
      toast.error('Neuropathy analysis failed: ' + error.message)
    }
  })

  const glucoseMutation = useMutation({
    mutationFn: () => mlApiService.predictions.predictGlucoseComplications(patientId, readingId),
    onSuccess: (data) => {
      toast.success('Glucose complications analysis completed')
      refetch()
    },
    onError: (error) => {
      toast.error('Glucose analysis failed: ' + error.message)
    }
  })

  const anomalyMutation = useMutation({
    mutationFn: () => mlApiService.predictions.detectAnomalies(patientId, readingId),
    onSuccess: (data) => {
      toast.success('Anomaly detection completed')
      refetch()
    },
    onError: (error) => {
      toast.error('Anomaly detection failed: ' + error.message)
    }
  })

  const riskMutation = useMutation({
    mutationFn: () => mlApiService.predictions.calculateRiskStratification(patientId, readingId),
    onSuccess: (data) => {
      toast.success('Risk stratification completed')
      refetch()
    },
    onError: (error) => {
      toast.error('Risk stratification failed: ' + error.message)
    }
  })

  const runAllAnalysis = async () => {
    setIsRunningAnalysis(true)
    try {
      await Promise.all([
        neuropathyMutation.mutateAsync(),
        glucoseMutation.mutateAsync(),
        anomalyMutation.mutateAsync(),
        riskMutation.mutateAsync()
      ])
      toast.success('All ML analyses completed successfully')
    } catch (error) {
      toast.error('Some analyses failed. Check individual results.')
    } finally {
      setIsRunningAnalysis(false)
    }
  }

  const getModelIcon = (modelType) => {
    switch (modelType) {
      case 'neuropathy_progression':
        return <Brain className="h-5 w-5" />
      case 'glucose_complications':
        return <Activity className="h-5 w-5" />
      case 'anomaly_detection':
        return <Zap className="h-5 w-5" />
      case 'risk_stratification':
        return <TrendingUp className="h-5 w-5" />
      default:
        return <BarChart3 className="h-5 w-5" />
    }
  }

  const getModelName = (modelType) => {
    switch (modelType) {
      case 'neuropathy_progression':
        return 'Neuropathy Progression'
      case 'glucose_complications':
        return 'Glucose Complications'
      case 'anomaly_detection':
        return 'Anomaly Detection'
      case 'risk_stratification':
        return 'Risk Stratification'
      default:
        return modelType
    }
  }

  const getRiskColor = (riskLevel) => {
    switch (riskLevel) {
      case 'HIGH':
        return 'text-red-600 bg-red-100'
      case 'MEDIUM':
        return 'text-yellow-600 bg-yellow-100'
      case 'LOW':
        return 'text-green-600 bg-green-100'
      default:
        return 'text-gray-600 bg-gray-100'
    }
  }

  const getConfidenceColor = (confidence) => {
    if (confidence > 0.8) return 'text-green-600'
    if (confidence > 0.6) return 'text-yellow-600'
    return 'text-red-600'
  }

  const formatPredictionValue = (value) => {
    return (value * 100).toFixed(1) + '%'
  }

  const formatConfidence = (confidence) => {
    return (confidence * 100).toFixed(1) + '%'
  }

  if (!canAccess('PROVIDER')) {
    return (
      <div className="text-center py-8">
        <AlertCircle className="h-12 w-12 text-yellow-500 mx-auto mb-4" />
        <p className="text-gray-600">Access to ML predictions requires provider privileges</p>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-semibold text-gray-900 flex items-center">
            <Brain className="h-6 w-6 text-primary-600 mr-2" />
            ML Predictions & Analysis
          </h2>
          <p className="text-gray-600 mt-1">
            AI-powered insights for neuropathy detection and risk assessment
          </p>
        </div>
        <div className="flex items-center space-x-3">
          <select
            value={selectedModel}
            onChange={(e) => setSelectedModel(e.target.value)}
            className="px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary-500"
          >
            <option value="all">All Models</option>
            <option value="neuropathy_progression">Neuropathy Progression</option>
            <option value="glucose_complications">Glucose Complications</option>
            <option value="anomaly_detection">Anomaly Detection</option>
            <option value="risk_stratification">Risk Stratification</option>
          </select>
          <button
            onClick={runAllAnalysis}
            disabled={isRunningAnalysis}
            className="flex items-center px-4 py-2 bg-primary-600 text-white rounded-md hover:bg-primary-700 focus:outline-none focus:ring-2 focus:ring-primary-500 disabled:opacity-50"
          >
            {isRunningAnalysis ? (
              <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
            ) : (
              <Brain className="h-4 w-4 mr-2" />
            )}
            Run Analysis
          </button>
        </div>
      </div>

      {/* Quick Actions */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <button
          onClick={() => neuropathyMutation.mutate()}
          disabled={neuropathyMutation.isLoading}
          className="p-4 bg-blue-50 hover:bg-blue-100 rounded-lg text-left transition-colors"
        >
          <Brain className="h-6 w-6 text-blue-600 mb-2" />
          <p className="font-medium text-gray-900">Neuropathy</p>
          <p className="text-sm text-gray-600">Progression analysis</p>
        </button>

        <button
          onClick={() => glucoseMutation.mutate()}
          disabled={glucoseMutation.isLoading}
          className="p-4 bg-green-50 hover:bg-green-100 rounded-lg text-left transition-colors"
        >
          <Activity className="h-6 w-6 text-green-600 mb-2" />
          <p className="font-medium text-gray-900">Glucose</p>
          <p className="text-sm text-gray-600">Complications risk</p>
        </button>

        <button
          onClick={() => anomalyMutation.mutate()}
          disabled={anomalyMutation.isLoading}
          className="p-4 bg-yellow-50 hover:bg-yellow-100 rounded-lg text-left transition-colors"
        >
          <Zap className="h-6 w-6 text-yellow-600 mb-2" />
          <p className="font-medium text-gray-900">Anomalies</p>
          <p className="text-sm text-gray-600">Sensor data analysis</p>
        </button>

        <button
          onClick={() => riskMutation.mutate()}
          disabled={riskMutation.isLoading}
          className="p-4 bg-purple-50 hover:bg-purple-100 rounded-lg text-left transition-colors"
        >
          <TrendingUp className="h-6 w-6 text-purple-600 mb-2" />
          <p className="font-medium text-gray-900">Risk</p>
          <p className="text-sm text-gray-600">Stratification</p>
        </button>
      </div>

      {/* Predictions List */}
      {isLoading ? (
        <div className="flex items-center justify-center py-8">
          <LoadingSpinner size="lg" />
        </div>
      ) : (
        <div className="space-y-4">
          {predictions?.length > 0 ? (
            predictions.map((prediction, index) => (
              <div key={index} className="bg-white rounded-lg border border-gray-200 p-6">
                <div className="flex items-start justify-between">
                  <div className="flex items-center space-x-3">
                    <div className="flex-shrink-0">
                      {getModelIcon(prediction.modelType)}
                    </div>
                    <div>
                      <h3 className="font-medium text-gray-900">
                        {getModelName(prediction.modelType)}
                      </h3>
                      <p className="text-sm text-gray-600">
                        {new Date(prediction.timestamp).toLocaleDateString()} at{' '}
                        {new Date(prediction.timestamp).toLocaleTimeString()}
                      </p>
                    </div>
                  </div>
                  <div className="text-right">
                    <div className={`inline-flex items-center px-3 py-1 rounded-full text-sm font-medium ${getRiskColor(prediction.riskLevel)}`}>
                      {prediction.riskLevel} Risk
                    </div>
                  </div>
                </div>

                <div className="mt-4 grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div className="bg-gray-50 rounded-lg p-4">
                    <div className="flex items-center">
                      <BarChart3 className="h-5 w-5 text-gray-500 mr-2" />
                      <span className="text-sm font-medium text-gray-700">Prediction</span>
                    </div>
                    <p className="text-2xl font-bold text-gray-900 mt-1">
                      {formatPredictionValue(prediction.predictionValue)}
                    </p>
                  </div>

                  <div className="bg-gray-50 rounded-lg p-4">
                    <div className="flex items-center">
                      <CheckCircle className={`h-5 w-5 mr-2 ${getConfidenceColor(prediction.confidence)}`} />
                      <span className="text-sm font-medium text-gray-700">Confidence</span>
                    </div>
                    <p className={`text-2xl font-bold mt-1 ${getConfidenceColor(prediction.confidence)}`}>
                      {formatConfidence(prediction.confidence)}
                    </p>
                  </div>

                  <div className="bg-gray-50 rounded-lg p-4">
                    <div className="flex items-center">
                      <Info className="h-5 w-5 text-gray-500 mr-2" />
                      <span className="text-sm font-medium text-gray-700">Model Version</span>
                    </div>
                    <p className="text-lg font-medium text-gray-900 mt-1">
                      {prediction.modelVersion || 'v1.0.0'}
                    </p>
                  </div>
                </div>

                {prediction.additionalData && (
                  <div className="mt-4 bg-blue-50 rounded-lg p-4">
                    <h4 className="font-medium text-blue-900 mb-2">Additional Insights</h4>
                    <div className="space-y-1 text-sm text-blue-800">
                      {Object.entries(JSON.parse(prediction.additionalData)).map(([key, value]) => (
                        <div key={key} className="flex justify-between">
                          <span className="capitalize">{key.replace(/_/g, ' ')}</span>
                          <span className="font-medium">{value}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            ))
          ) : (
            <div className="text-center py-8">
              <Brain className="h-12 w-12 text-gray-400 mx-auto mb-4" />
              <p className="text-gray-600">No ML predictions available</p>
              <p className="text-sm text-gray-500 mt-1">
                Run an analysis to generate AI-powered insights
              </p>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

export default MLPredictions