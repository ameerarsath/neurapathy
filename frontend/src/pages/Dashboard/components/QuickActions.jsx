import { useState } from 'react'
import { motion } from 'framer-motion'
import { 
  Plus, 
  Download, 
  RefreshCw, 
  Settings, 
  Calendar, 
  User, 
  BarChart3, 
  Bell,
  Smartphone,
  FileText,
  Video,
  MessageSquare,
  Shield,
  HelpCircle
} from 'lucide-react'
import Card from '@components/common/Card'
import Button from '@components/common/Button'
import { useNavigate } from 'react-router-dom'

function QuickActions({ user, onRefresh, onExport }) {
  const navigate = useNavigate()
  const [hoveredAction, setHoveredAction] = useState(null)

  const getActionsForRole = (userRole) => {
    const baseActions = [
      {
        id: 'new-test',
        label: 'Start Test',
        icon: Plus,
        color: 'bg-blue-600 hover:bg-blue-700',
        textColor: 'text-white',
        action: () => navigate('/test-sessions'),
        description: 'Begin a new neuropathy assessment'
      },
      {
        id: 'view-results',
        label: 'View Results',
        icon: BarChart3,
        color: 'bg-green-600 hover:bg-green-700',
        textColor: 'text-white',
        action: () => navigate('/test-results'),
        description: 'Review test results and analytics'
      },
      {
        id: 'device-status',
        label: 'Devices',
        icon: Smartphone,
        color: 'bg-purple-600 hover:bg-purple-700',
        textColor: 'text-white',
        action: () => navigate('/devices'),
        description: 'Manage connected devices'
      },
      {
        id: 'schedule',
        label: 'Schedule',
        icon: Calendar,
        color: 'bg-orange-600 hover:bg-orange-700',
        textColor: 'text-white',
        action: () => navigate('/appointments'),
        description: 'View and manage appointments'
      }
    ]

    const patientActions = [
      ...baseActions,
      {
        id: 'profile',
        label: 'My Profile',
        icon: User,
        color: 'bg-gray-600 hover:bg-gray-700',
        textColor: 'text-white',
        action: () => navigate('/patients/' + user?.id),
        description: 'Update personal information'
      },
      {
        id: 'telemedicine',
        label: 'Consult',
        icon: Video,
        color: 'bg-indigo-600 hover:bg-indigo-700',
        textColor: 'text-white',
        action: () => navigate('/telemedicine'),
        description: 'Start video consultation'
      }
    ]

    const providerActions = [
      ...baseActions,
      {
        id: 'patients',
        label: 'Patients',
        icon: User,
        color: 'bg-teal-600 hover:bg-teal-700',
        textColor: 'text-white',
        action: () => navigate('/patients'),
        description: 'Manage patient records'
      },
      {
        id: 'ml-predictions',
        label: 'ML Analysis',
        icon: BarChart3,
        color: 'bg-pink-600 hover:bg-pink-700',
        textColor: 'text-white',
        action: () => navigate('/ml-predictions'),
        description: 'Review ML predictions'
      },
      {
        id: 'reports',
        label: 'Reports',
        icon: FileText,
        color: 'bg-yellow-600 hover:bg-yellow-700',
        textColor: 'text-white',
        action: () => navigate('/reports'),
        description: 'Generate medical reports'
      }
    ]

    const adminActions = [
      {
        id: 'system-health',
        label: 'System Health',
        icon: Shield,
        color: 'bg-red-600 hover:bg-red-700',
        textColor: 'text-white',
        action: () => navigate('/system-health'),
        description: 'Monitor system status'
      },
      {
        id: 'user-management',
        label: 'Users',
        icon: User,
        color: 'bg-gray-600 hover:bg-gray-700',
        textColor: 'text-white',
        action: () => navigate('/user-management'),
        description: 'Manage user accounts'
      },
      {
        id: 'audit-logs',
        label: 'Audit Logs',
        icon: FileText,
        color: 'bg-blue-600 hover:bg-blue-700',
        textColor: 'text-white',
        action: () => navigate('/audit-logs'),
        description: 'Review system audit logs'
      },
      {
        id: 'settings',
        label: 'Settings',
        icon: Settings,
        color: 'bg-gray-600 hover:bg-gray-700',
        textColor: 'text-white',
        action: () => navigate('/settings'),
        description: 'Configure system settings'
      }
    ]

    switch (userRole) {
      case 'ADMIN':
        return [...providerActions.slice(0, 4), ...adminActions]
      case 'PROVIDER':
        return providerActions
      case 'CAREGIVER':
        return [...patientActions.slice(0, 5), {
          id: 'support',
          label: 'Support',
          icon: MessageSquare,
          color: 'bg-green-600 hover:bg-green-700',
          textColor: 'text-white',
          action: () => navigate('/support'),
          description: 'Get help and support'
        }]
      case 'PATIENT':
      default:
        return patientActions
    }
  }

  const actions = getActionsForRole(user?.role)
  const displayActions = actions.slice(0, 6) // Show max 6 actions

  const utilityActions = [
    {
      id: 'refresh',
      label: 'Refresh',
      icon: RefreshCw,
      action: onRefresh,
      variant: 'outline'
    },
    {
      id: 'export',
      label: 'Export',
      icon: Download,
      action: onExport,
      variant: 'outline'
    },
    {
      id: 'help',
      label: 'Help',
      icon: HelpCircle,
      action: () => navigate('/help'),
      variant: 'outline'
    }
  ]

  return (
    <Card title="Quick Actions" className="h-full">
      <div className="space-y-4">
        {/* Primary Actions Grid */}
        <div className="grid grid-cols-2 gap-2">
          {displayActions.map((action, index) => {
            const Icon = action.icon
            
            return (
              <motion.button
                key={action.id}
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: index * 0.05 }}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                onHoverStart={() => setHoveredAction(action.id)}
                onHoverEnd={() => setHoveredAction(null)}
                onClick={action.action}
                className={`p-3 rounded-lg ${action.color} ${action.textColor} transition-all duration-200 group relative overflow-hidden`}
              >
                {/* Background Gradient */}
                <div className="absolute inset-0 bg-gradient-to-r from-white/10 to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />
                
                <div className="relative flex flex-col items-center gap-2">
                  <Icon className="w-5 h-5" />
                  <span className="text-xs font-medium">{action.label}</span>
                </div>

                {/* Tooltip */}
                {hoveredAction === action.id && (
                  <motion.div
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="absolute bottom-full left-1/2 transform -translate-x-1/2 mb-2 px-2 py-1 bg-gray-900 text-white text-xs rounded whitespace-nowrap z-10"
                  >
                    {action.description}
                    <div className="absolute top-full left-1/2 transform -translate-x-1/2 w-0 h-0 border-l-4 border-r-4 border-t-4 border-transparent border-t-gray-900" />
                  </motion.div>
                )}
              </motion.button>
            )
          })}
        </div>

        {/* Utility Actions */}
        <div className="pt-3 border-t border-gray-200 dark:border-gray-700">
          <div className="flex justify-between gap-1">
            {utilityActions.map((action, index) => {
              const Icon = action.icon
              
              return (
                <Button
                  key={action.id}
                  variant={action.variant}
                  size="sm"
                  onClick={action.action}
                  className="flex-1 flex flex-col items-center gap-1 h-auto py-2"
                >
                  <Icon className="w-4 h-4" />
                  <span className="text-xs">{action.label}</span>
                </Button>
              )
            })}
          </div>
        </div>

        {/* Recent Activity Quick Stats */}
        <div className="bg-gray-50 dark:bg-gray-800/50 rounded-lg p-3">
          <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            Today's Summary
          </h4>
          <div className="grid grid-cols-3 gap-2 text-center text-xs">
            <div>
              <div className="font-medium text-blue-600">
                {user?.testsToday || 0}
              </div>
              <div className="text-gray-500">Tests</div>
            </div>
            <div>
              <div className="font-medium text-green-600">
                {user?.deviceConnections || 1}
              </div>
              <div className="text-gray-500">Devices</div>
            </div>
            <div>
              <div className="font-medium text-purple-600">
                {user?.alertsToday || 0}
              </div>
              <div className="text-gray-500">Alerts</div>
            </div>
          </div>
        </div>

        {/* Emergency Contact (for patients/caregivers) */}
        {(user?.role === 'PATIENT' || user?.role === 'CAREGIVER') && (
          <Button
            variant="outline"
            size="sm"
            className="w-full border-red-300 text-red-600 hover:bg-red-50 dark:hover:bg-red-900/20"
            onClick={() => navigate('/emergency')}
          >
            <Bell className="w-4 h-4 mr-2" />
            Emergency Contact
          </Button>
        )}

        {/* Settings Access */}
        <Button
          variant="ghost"
          size="sm"
          className="w-full"
          onClick={() => navigate('/settings')}
        >
          <Settings className="w-4 h-4 mr-2" />
          Settings & Preferences
        </Button>
      </div>
    </Card>
  )
}

export default QuickActions