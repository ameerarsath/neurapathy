import React from 'react'
import { 
  Server, 
  Database, 
  Wifi, 
  Shield, 
  AlertTriangle,
  CheckCircle,
  Clock
} from 'lucide-react'

const SystemStatus = ({ criticalAlerts = 0, lowBatteryDevices = 0 }) => {
  const systemComponents = [
    {
      name: 'API Server',
      status: 'healthy',
      icon: Server,
      details: 'Response time: 45ms'
    },
    {
      name: 'Database',
      status: 'healthy',
      icon: Database,
      details: 'Connections: 12/100'
    },
    {
      name: 'Device Network',
      status: 'healthy',
      icon: Wifi,
      details: '45 devices connected'
    },
    {
      name: 'Security',
      status: 'healthy',
      icon: Shield,
      details: 'All systems secure'
    }
  ]

  const alerts = [
    ...(criticalAlerts > 0 ? [{
      level: 'critical',
      message: `${criticalAlerts} critical medical readings require attention`,
      icon: AlertTriangle
    }] : []),
    ...(lowBatteryDevices > 0 ? [{
      level: 'warning',
      message: `${lowBatteryDevices} devices have low battery`,
      icon: AlertTriangle
    }] : [])
  ]

  const getStatusColor = (status) => {
    const colors = {
      healthy: 'text-green-600 bg-green-50',
      warning: 'text-yellow-600 bg-yellow-50',
      error: 'text-red-600 bg-red-50'
    }
    return colors[status] || colors.healthy
  }

  const getStatusIcon = (status) => {
    return status === 'healthy' ? CheckCircle : AlertTriangle
  }

  return (
    <div className="medical-card">
      <h3 className="text-lg font-medium text-neutral-900 mb-4">
        System Status
      </h3>

      {/* System Components */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
        {systemComponents.map((component) => {
          const Icon = component.icon
          const StatusIcon = getStatusIcon(component.status)
          
          return (
            <div key={component.name} className="border border-neutral-200 rounded-lg p-4">
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center">
                  <Icon className="h-4 w-4 text-neutral-600 mr-2" />
                  <span className="text-sm font-medium text-neutral-900">
                    {component.name}
                  </span>
                </div>
                <StatusIcon className={`h-4 w-4 ${component.status === 'healthy' ? 'text-green-600' : 'text-red-600'}`} />
              </div>
              <p className="text-xs text-neutral-500">{component.details}</p>
            </div>
          )
        })}
      </div>

      {/* Active Alerts */}
      {alerts.length > 0 && (
        <div className="space-y-3">
          <h4 className="text-sm font-medium text-neutral-900">Active Alerts</h4>
          {alerts.map((alert, index) => {
            const AlertIcon = alert.icon
            const colorClass = alert.level === 'critical' ? 'border-red-200 bg-red-50' : 'border-yellow-200 bg-yellow-50'
            const iconColor = alert.level === 'critical' ? 'text-red-600' : 'text-yellow-600'
            
            return (
              <div key={index} className={`border rounded-lg p-3 ${colorClass}`}>
                <div className="flex items-center">
                  <AlertIcon className={`h-4 w-4 mr-2 ${iconColor}`} />
                  <span className="text-sm text-neutral-900">{alert.message}</span>
                </div>
              </div>
            )
          })}
        </div>
      )}

      {/* No Alerts State */}
      {alerts.length === 0 && (
        <div className="text-center py-4">
          <CheckCircle className="h-8 w-8 text-green-600 mx-auto mb-2" />
          <p className="text-sm text-neutral-600">All systems operational</p>
          <p className="text-xs text-neutral-500">No alerts or warnings</p>
        </div>
      )}

      {/* Last Updated */}
      <div className="mt-4 pt-4 border-t border-neutral-200">
        <div className="flex items-center justify-between text-xs text-neutral-500">
          <div className="flex items-center">
            <Clock className="h-3 w-3 mr-1" />
            <span>Last updated: Just now</span>
          </div>
          <button className="text-primary-600 hover:text-primary-700">
            Refresh
          </button>
        </div>
      </div>
    </div>
  )
}

export default SystemStatus