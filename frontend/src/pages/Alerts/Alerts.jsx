import { useState } from 'react'
import { useQuery } from 'react-query'
import { motion } from 'framer-motion'
import { Helmet } from 'react-helmet-async'
import { 
  Bell, 
  AlertTriangle, 
  Info, 
  CheckCircle,
  X,
  Filter,
  Settings,
  Clock
} from 'lucide-react'

import { useAuth } from '@contexts/AuthContext'
import { api } from '@services/api'
import Card from '@components/common/Card'
import Button from '@components/common/Button'
import LoadingSpinner from '@components/common/LoadingSpinner'
import { getTimeAgo } from '@utils/dateUtils'

function Alerts() {
  const [filter, setFilter] = useState('all')
  const { user } = useAuth()

  const { data: alertsResponse, isLoading, error } = useQuery(
    ['alerts', user?.id, filter],
    () => api.alert.getAlerts({ patientId: user?.id, status: filter }),
    {
      enabled: !!user?.id,
      staleTime: 30 * 1000,
    }
  )

  const alerts = alertsResponse?.data || []

  if (isLoading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <LoadingSpinner size="lg" text="Loading alerts..." />
      </div>
    )
  }

  if (error) {
    console.warn('Failed to load alerts from API, using mock data:', error)
  }

  const getAlertIcon = (severity) => {
    switch (severity?.toLowerCase()) {
      case 'critical': return <AlertTriangle className="w-5 h-5 text-red-600" />
      case 'warning': return <AlertTriangle className="w-5 h-5 text-yellow-600" />
      case 'info': return <Info className="w-5 h-5 text-blue-600" />
      default: return <Bell className="w-5 h-5 text-gray-600" />
    }
  }

  const getAlertColor = (severity) => {
    switch (severity?.toLowerCase()) {
      case 'critical': return 'border-l-red-500 bg-red-50 dark:bg-red-900/20'
      case 'warning': return 'border-l-yellow-500 bg-yellow-50 dark:bg-yellow-900/20'
      case 'info': return 'border-l-blue-500 bg-blue-50 dark:bg-blue-900/20'
      default: return 'border-l-gray-500 bg-gray-50 dark:bg-gray-900/20'
    }
  }

  // Use only API data - no mock fallback
  const displayAlerts = alerts || []

  return (
    <div className="space-y-6">
      <Helmet>
        <title>Alerts - Smart Shoe Monitor</title>
        <meta name="description" content="View and manage your health alerts and notifications" />
      </Helmet>

      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="flex flex-col lg:flex-row lg:items-center lg:justify-between"
      >
        <div>
          <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
            Alerts & Notifications
          </h1>
          <p className="mt-2 text-gray-600 dark:text-gray-400">
            Stay informed about your health status and device updates
          </p>
        </div>
        
        <div className="mt-4 lg:mt-0 flex gap-2">
          <Button variant="outline" size="sm">
            <Settings className="w-4 h-4 mr-2" />
            Settings
          </Button>
          <Button variant="outline" size="sm">
            <CheckCircle className="w-4 h-4 mr-2" />
            Mark All Read
          </Button>
        </div>
      </motion.div>

      {/* Alert Stats */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="grid grid-cols-1 md:grid-cols-4 gap-4"
      >
        <Card className="p-4">
          <div className="flex items-center gap-3">
            <AlertTriangle className="w-5 h-5 text-red-600" />
            <div>
              <div className="text-lg font-semibold text-gray-900 dark:text-white">
                {displayAlerts.filter(a => a.severity === 'critical').length}
              </div>
              <div className="text-sm text-gray-600 dark:text-gray-400">
                Critical Alerts
              </div>
            </div>
          </div>
        </Card>
        
        <Card className="p-4">
          <div className="flex items-center gap-3">
            <AlertTriangle className="w-5 h-5 text-yellow-600" />
            <div>
              <div className="text-lg font-semibold text-gray-900 dark:text-white">
                {displayAlerts.filter(a => a.severity === 'warning').length}
              </div>
              <div className="text-sm text-gray-600 dark:text-gray-400">
                Warnings
              </div>
            </div>
          </div>
        </Card>
        
        <Card className="p-4">
          <div className="flex items-center gap-3">
            <Bell className="w-5 h-5 text-blue-600" />
            <div>
              <div className="text-lg font-semibold text-gray-900 dark:text-white">
                {displayAlerts.filter(a => !a.read).length}
              </div>
              <div className="text-sm text-gray-600 dark:text-gray-400">
                Unread
              </div>
            </div>
          </div>
        </Card>
        
        <Card className="p-4">
          <div className="flex items-center gap-3">
            <Clock className="w-5 h-5 text-gray-600" />
            <div>
              <div className="text-lg font-semibold text-gray-900 dark:text-white">
                {displayAlerts.length}
              </div>
              <div className="text-sm text-gray-600 dark:text-gray-400">
                Total Alerts
              </div>
            </div>
          </div>
        </Card>
      </motion.div>

      {/* Filter Tabs */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
        className="flex items-center gap-1 bg-gray-100 dark:bg-gray-800 rounded-lg p-1"
      >
        {[
          { value: 'all', label: 'All' },
          { value: 'unread', label: 'Unread' },
          { value: 'critical', label: 'Critical' },
          { value: 'warning', label: 'Warning' },
          { value: 'info', label: 'Info' }
        ].map((tab) => (
          <button
            key={tab.value}
            onClick={() => setFilter(tab.value)}
            className={`px-4 py-2 text-sm font-medium rounded-md transition-colors ${
              filter === tab.value
                ? 'bg-white dark:bg-gray-700 text-gray-900 dark:text-white shadow-sm'
                : 'text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white'
            }`}
          >
            {tab.label}
          </button>
        ))}
      </motion.div>

      {/* Alerts List */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
      >
        <Card title="Recent Alerts">
          <div className="space-y-3">
            {displayAlerts.map((alert, index) => (
              <motion.div
                key={alert.id}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: index * 0.1 }}
                className={`p-4 border-l-4 rounded-lg ${getAlertColor(alert.severity)} ${
                  !alert.read ? 'ring-1 ring-blue-200 dark:ring-blue-800' : ''
                }`}
              >
                <div className="flex items-start justify-between">
                  <div className="flex items-start gap-3 flex-1">
                    {getAlertIcon(alert.severity)}
                    <div className="flex-1">
                      <div className="flex items-center gap-2 mb-1">
                        <h3 className="font-medium text-gray-900 dark:text-white">
                          {alert.title}
                        </h3>
                        {!alert.read && (
                          <span className="w-2 h-2 bg-blue-600 rounded-full"></span>
                        )}
                      </div>
                      <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                        {alert.message}
                      </p>
                      <div className="flex items-center gap-2 text-xs text-gray-500 dark:text-gray-400">
                        <Clock className="w-3 h-3" />
                        <span>{getTimeAgo(alert.timestamp)}</span>
                      </div>
                    </div>
                  </div>
                  
                  <div className="flex items-center gap-2 ml-4">
                    <Button variant="ghost" size="sm">
                      <CheckCircle className="w-4 h-4" />
                    </Button>
                    <Button variant="ghost" size="sm">
                      <X className="w-4 h-4" />
                    </Button>
                  </div>
                </div>
              </motion.div>
            ))}
          </div>
        </Card>
      </motion.div>

      {/* Alert Settings */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.3 }}
      >
        <Card title="Alert Preferences">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <h4 className="font-medium text-gray-900 dark:text-white mb-3">
                Notification Methods
              </h4>
              <div className="space-y-3">
                <label className="flex items-center">
                  <input type="checkbox" defaultChecked className="mr-2" />
                  <span className="text-sm text-gray-700 dark:text-gray-300">Email notifications</span>
                </label>
                <label className="flex items-center">
                  <input type="checkbox" defaultChecked className="mr-2" />
                  <span className="text-sm text-gray-700 dark:text-gray-300">SMS alerts</span>
                </label>
                <label className="flex items-center">
                  <input type="checkbox" defaultChecked className="mr-2" />
                  <span className="text-sm text-gray-700 dark:text-gray-300">Push notifications</span>
                </label>
              </div>
            </div>
            
            <div>
              <h4 className="font-medium text-gray-900 dark:text-white mb-3">
                Alert Types
              </h4>
              <div className="space-y-3">
                <label className="flex items-center">
                  <input type="checkbox" defaultChecked className="mr-2" />
                  <span className="text-sm text-gray-700 dark:text-gray-300">Test result alerts</span>
                </label>
                <label className="flex items-center">
                  <input type="checkbox" defaultChecked className="mr-2" />
                  <span className="text-sm text-gray-700 dark:text-gray-300">Device status updates</span>
                </label>
                <label className="flex items-center">
                  <input type="checkbox" className="mr-2" />
                  <span className="text-sm text-gray-700 dark:text-gray-300">Appointment reminders</span>
                </label>
              </div>
            </div>
          </div>
        </Card>
      </motion.div>
    </div>
  )
}

export default Alerts