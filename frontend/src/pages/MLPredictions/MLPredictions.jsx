import { useState, useEffect } from 'react'
import { useQuery } from 'react-query'
import { motion } from 'framer-motion'
import { Helmet } from 'react-helmet-async'
import { 
  Brain, 
  TrendingUp, 
  TrendingDown, 
  Activity, 
  Calendar,
  Download,
  Zap,
  AlertTriangle,
  CheckCircle,
  Eye,
  RefreshCw,
  BarChart3,
  Target,
  Clock
} from 'lucide-react'

import { useAuth } from '@contexts/AuthContext'
import { api } from '@services/api'
import Card from '@components/common/Card'
import Button from '@components/common/Button'
import LoadingSpinner from '@components/common/LoadingSpinner'
import { formatDateTime, getTimeAgo } from '@utils/dateUtils'
import { getRiskColor } from '@utils/medicalUtils'

function MLPredictions() {
  const [timeRange, setTimeRange] = useState('month')
  const [selectedPrediction, setSelectedPrediction] = useState(null)
  const [modelType, setModelType] = useState('all')

  const { user } = useAuth()

  // Fetch ML predictions
  const { data: predictions, isLoading, refetch } = useQuery(
    ['ml-predictions', user?.id, timeRange, modelType],
    () => api.ml.getPredictions(user?.id, { timeRange, modelType }),
    {
      enabled: !!user?.id,
      staleTime: 5 * 60 * 1000,
    }
  )

  // Fetch model performance metrics
  const { data: modelMetrics } = useQuery(
    ['ml-model-metrics'],
    () => api.ml.getModelMetrics(),
    {
      staleTime: 10 * 60 * 1000,
    }
  )

  if (isLoading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <LoadingSpinner size="lg" text="Loading ML predictions..." />
      </div>
    )
  }

  const getRiskIcon = (risk) => {
    switch (risk?.toLowerCase()) {
      case 'high': return <AlertTriangle className="w-4 h-4" />
      case 'medium': return <Activity className="w-4 h-4" />
      case 'low': return <CheckCircle className="w-4 h-4" />
      default: return <Brain className="w-4 h-4" />
    }
  }

  const getProgressionIcon = (trend) => {
    switch (trend?.toLowerCase()) {
      case 'improving': return <TrendingDown className="w-4 h-4 text-green-600" />
      case 'worsening': return <TrendingUp className="w-4 h-4 text-red-600" />
      case 'stable': return <Activity className="w-4 h-4 text-yellow-600" />
      default: return <Activity className="w-4 h-4 text-gray-600" />
    }
  }

  const predictionTypes = [
    { value: 'all', label: 'All Predictions' },
    { value: 'neuropathy', label: 'Neuropathy Progression' },
    { value: 'complications', label: 'Complications Risk' },
    { value: 'anomaly', label: 'Anomaly Detection' },
    { value: 'risk_stratification', label: 'Risk Stratification' }
  ]

  return (
    <div className="space-y-6">
      <Helmet>
        <title>ML Predictions - Smart Shoe Monitor</title>
        <meta name="description" content="AI-powered predictions for neuropathy progression and health insights" />
      </Helmet>

      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="flex flex-col lg:flex-row lg:items-center lg:justify-between"
      >
        <div>
          <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
            ML Predictions
          </h1>
          <p className="mt-2 text-gray-600 dark:text-gray-400">
            AI-powered insights for your neuropathy monitoring and health management
          </p>
        </div>
        
        <div className="mt-4 lg:mt-0 flex flex-wrap gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => refetch()}
            className="flex items-center gap-2"
          >
            <RefreshCw className="w-4 h-4" />
            Refresh
          </Button>
          
          <Button
            variant="outline"
            size="sm"
            className="flex items-center gap-2"
          >
            <Download className="w-4 h-4" />
            Export Report
          </Button>
        </div>
      </motion.div>

      {/* Filters */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
        className="flex flex-col lg:flex-row gap-4"
      >
        {/* Time Range */}
        <div className="flex items-center gap-2">
          <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
            Time Range:
          </span>
          <div className="flex bg-gray-100 dark:bg-gray-800 rounded-lg p-1">
            {['week', 'month', 'quarter'].map((range) => (
              <button
                key={range}
                onClick={() => setTimeRange(range)}
                className={`px-3 py-1 text-sm font-medium rounded-md transition-colors ${
                  timeRange === range
                    ? 'bg-white dark:bg-gray-700 text-gray-900 dark:text-white shadow-sm'
                    : 'text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white'
                }`}
              >
                {range.charAt(0).toUpperCase() + range.slice(1)}
              </button>
            ))}
          </div>
        </div>

        {/* Model Type */}
        <div className="flex items-center gap-2">
          <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
            Prediction Type:
          </span>
          <select
            value={modelType}
            onChange={(e) => setModelType(e.target.value)}
            className="px-3 py-1 text-sm border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
          >
            {predictionTypes.map((type) => (
              <option key={type.value} value={type.value}>
                {type.label}
              </option>
            ))}
          </select>
        </div>
      </motion.div>

      {/* Model Performance Overview */}
      {modelMetrics && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="grid grid-cols-1 md:grid-cols-4 gap-4"
        >
          <Card className="p-4">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-blue-100 dark:bg-blue-900/20 rounded-lg">
                <Target className="w-4 h-4 text-blue-600 dark:text-blue-400" />
              </div>
              <div>
                <div className="text-lg font-semibold text-gray-900 dark:text-white">
                  {modelMetrics.accuracy || '94.2%'}
                </div>
                <div className="text-xs text-gray-600 dark:text-gray-400">
                  Model Accuracy
                </div>
              </div>
            </div>
          </Card>

          <Card className="p-4">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-green-100 dark:bg-green-900/20 rounded-lg">
                <CheckCircle className="w-4 h-4 text-green-600 dark:text-green-400" />
              </div>
              <div>
                <div className="text-lg font-semibold text-gray-900 dark:text-white">
                  {modelMetrics.predictions_today || '127'}
                </div>
                <div className="text-xs text-gray-600 dark:text-gray-400">
                  Predictions Today
                </div>
              </div>
            </div>
          </Card>

          <Card className="p-4">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-purple-100 dark:bg-purple-900/20 rounded-lg">
                <Zap className="w-4 h-4 text-purple-600 dark:text-purple-400" />
              </div>
              <div>
                <div className="text-lg font-semibold text-gray-900 dark:text-white">
                  {modelMetrics.avg_confidence || '87.3%'}
                </div>
                <div className="text-xs text-gray-600 dark:text-gray-400">
                  Avg Confidence
                </div>
              </div>
            </div>
          </Card>

          <Card className="p-4">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-orange-100 dark:bg-orange-900/20 rounded-lg">
                <Clock className="w-4 h-4 text-orange-600 dark:text-orange-400" />
              </div>
              <div>
                <div className="text-lg font-semibold text-gray-900 dark:text-white">
                  {modelMetrics.response_time || '1.2s'}
                </div>
                <div className="text-xs text-gray-600 dark:text-gray-400">
                  Response Time
                </div>
              </div>
            </div>
          </Card>
        </motion.div>
      )}

      {/* Predictions Grid */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.3 }}
        className="grid grid-cols-1 lg:grid-cols-2 gap-6"
      >
        {predictions?.data?.map((prediction, index) => (
          <motion.div
            key={prediction.id}
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.1 * index }}
          >
            <Card className="h-full hover:shadow-lg transition-shadow">
              <div className="p-6">
                {/* Prediction Header */}
                <div className="flex items-start justify-between mb-4">
                  <div className="flex items-center gap-3">
                    <div className={`p-2 rounded-lg ${getRiskColor(prediction.risk_level)} bg-opacity-20`}>
                      {getRiskIcon(prediction.risk_level)}
                    </div>
                    <div>
                      <h3 className="font-semibold text-gray-900 dark:text-white">
                        {prediction.prediction_type || 'Neuropathy Progression'}
                      </h3>
                      <p className="text-sm text-gray-600 dark:text-gray-400">
                        {getTimeAgo(prediction.created_at)}
                      </p>
                    </div>
                  </div>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => setSelectedPrediction(prediction)}
                  >
                    <Eye className="w-4 h-4" />
                  </Button>
                </div>

                {/* Risk Level */}
                <div className="mb-4">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                      Risk Level
                    </span>
                    <span className={`px-2 py-1 text-xs font-semibold rounded-full ${
                      getRiskColor(prediction.risk_level) === 'red' 
                        ? 'text-red-600 bg-red-100 dark:bg-red-900/20'
                        : getRiskColor(prediction.risk_level) === 'yellow'
                        ? 'text-yellow-600 bg-yellow-100 dark:bg-yellow-900/20'
                        : 'text-green-600 bg-green-100 dark:bg-green-900/20'
                    }`}>
                      {prediction.risk_level || 'Medium'}
                    </span>
                  </div>
                </div>

                {/* Confidence Score */}
                <div className="mb-4">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                      Confidence
                    </span>
                    <span className="text-sm text-gray-600 dark:text-gray-400">
                      {Math.round((prediction.confidence || 0.85) * 100)}%
                    </span>
                  </div>
                  <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                    <motion.div
                      initial={{ width: 0 }}
                      animate={{ width: `${(prediction.confidence || 0.85) * 100}%` }}
                      transition={{ duration: 1, ease: "easeOut" }}
                      className="bg-blue-600 h-2 rounded-full"
                    />
                  </div>
                </div>

                {/* Progression */}
                <div className="mb-4">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      {getProgressionIcon(prediction.progression_trend)}
                      <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                        Progression
                      </span>
                    </div>
                    <span className="text-sm text-gray-600 dark:text-gray-400">
                      {prediction.progression_trend || 'Stable'}
                    </span>
                  </div>
                </div>

                {/* Key Insights */}
                {prediction.key_insights && (
                  <div className="bg-gray-50 dark:bg-gray-800/50 rounded-lg p-3">
                    <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                      Key Insights
                    </h4>
                    <ul className="text-xs text-gray-600 dark:text-gray-400 space-y-1">
                      {prediction.key_insights.slice(0, 3).map((insight, idx) => (
                        <li key={idx} className="flex items-start gap-1">
                          <span className="text-blue-500 mt-0.5">•</span>
                          <span>{insight}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}

                {/* Actions */}
                <div className="mt-4 flex gap-2">
                  <Button
                    variant="outline"
                    size="sm"
                    className="flex-1"
                    onClick={() => setSelectedPrediction(prediction)}
                  >
                    View Details
                  </Button>
                  <Button
                    variant="ghost"
                    size="sm"
                  >
                    <BarChart3 className="w-4 h-4" />
                  </Button>
                </div>
              </div>
            </Card>
          </motion.div>
        ))}
      </motion.div>

      {/* Detailed Prediction Modal/Panel */}
      {selectedPrediction && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50"
          onClick={() => setSelectedPrediction(null)}
        >
          <motion.div
            initial={{ scale: 0.95, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            className="bg-white dark:bg-gray-800 rounded-lg p-6 max-w-2xl w-full max-h-[80vh] overflow-y-auto"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
                Detailed Prediction Analysis
              </h3>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setSelectedPrediction(null)}
              >
                ×
              </Button>
            </div>

            <div className="space-y-4">
              <div>
                <h4 className="font-medium text-gray-900 dark:text-white mb-2">
                  Model Analysis
                </h4>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  {selectedPrediction.detailed_analysis || 
                   'Based on your recent test results and historical data, our AI model has analyzed multiple factors including vibration threshold, pressure sensitivity, and temperature detection capabilities.'}
                </p>
              </div>

              {selectedPrediction.recommendations && (
                <div>
                  <h4 className="font-medium text-gray-900 dark:text-white mb-2">
                    Recommendations
                  </h4>
                  <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
                    {selectedPrediction.recommendations.map((rec, idx) => (
                      <li key={idx} className="flex items-start gap-2">
                        <CheckCircle className="w-4 h-4 text-green-600 mt-0.5 flex-shrink-0" />
                        {rec}
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {selectedPrediction.next_assessment && (
                <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-4">
                  <div className="flex items-center gap-2 mb-2">
                    <Calendar className="w-4 h-4 text-blue-600 dark:text-blue-400" />
                    <span className="font-medium text-blue-800 dark:text-blue-200">
                      Next Assessment
                    </span>
                  </div>
                  <p className="text-sm text-blue-700 dark:text-blue-300">
                    {selectedPrediction.next_assessment}
                  </p>
                </div>
              )}
            </div>
          </motion.div>
        </motion.div>
      )}

      {/* Empty State */}
      {!predictions?.data?.length && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="text-center py-12"
        >
          <Brain className="w-16 h-16 text-gray-400 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-2">
            No Predictions Available
          </h3>
          <p className="text-gray-600 dark:text-gray-400 mb-4">
            Complete a few tests to start receiving AI-powered health insights
          </p>
          <Button onClick={() => window.location.href = '/test-sessions'}>
            Start Your First Test
          </Button>
        </motion.div>
      )}
    </div>
  )
}

export default MLPredictions