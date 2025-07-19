import React from 'react'
import { useAuth } from '../../contexts/AuthContext'
import { Link } from 'react-router-dom'
import { 
  UserPlus, 
  Smartphone, 
  Activity, 
  AlertTriangle,
  Settings,
  BarChart3
} from 'lucide-react'

const QuickActions = () => {
  const { canAccess } = useAuth()

  const actions = [
    {
      title: 'Add Patient',
      description: 'Register new patient',
      icon: UserPlus,
      href: '/patients?action=add',
      color: 'primary',
      roles: ['ADMIN', 'PROVIDER']
    },
    {
      title: 'Pair Device',
      description: 'Connect smart shoe',
      icon: Smartphone,
      href: '/devices?action=pair',
      color: 'secondary',
      roles: ['ADMIN', 'PROVIDER']
    },
    {
      title: 'View Readings',
      description: 'Medical data analysis',
      icon: Activity,
      href: '/medical-readings',
      color: 'success',
      roles: ['ADMIN', 'PROVIDER', 'PATIENT']
    },
    {
      title: 'Critical Alerts',
      description: 'Urgent attention needed',
      icon: AlertTriangle,
      href: '/medical-readings?filter=critical',
      color: 'error',
      roles: ['ADMIN', 'PROVIDER']
    },
    {
      title: 'Reports',
      description: 'Generate analytics',
      icon: BarChart3,
      href: '/reports',
      color: 'warning',
      roles: ['ADMIN', 'PROVIDER']
    },
    {
      title: 'Settings',
      description: 'System configuration',
      icon: Settings,
      href: '/settings',
      color: 'neutral',
      roles: ['ADMIN', 'PROVIDER', 'PATIENT', 'USER']
    }
  ]

  const filteredActions = actions.filter(action => 
    action.roles.some(role => canAccess(role))
  )

  return (
    <div className="medical-card">
      <h3 className="text-lg font-medium text-neutral-900 mb-4">
        Quick Actions
      </h3>
      <div className="space-y-3">
        {filteredActions.map((action) => {
          const Icon = action.icon
          const colorClasses = {
            primary: 'text-primary-600 bg-primary-50 hover:bg-primary-100',
            secondary: 'text-secondary-600 bg-secondary-50 hover:bg-secondary-100',
            success: 'text-green-600 bg-green-50 hover:bg-green-100',
            warning: 'text-yellow-600 bg-yellow-50 hover:bg-yellow-100',
            error: 'text-red-600 bg-red-50 hover:bg-red-100',
            neutral: 'text-neutral-600 bg-neutral-50 hover:bg-neutral-100'
          }

          return (
            <Link
              key={action.title}
              to={action.href}
              className="flex items-center p-3 rounded-lg border border-neutral-200 hover:border-neutral-300 transition-colors group"
            >
              <div className={`p-2 rounded-md ${colorClasses[action.color]} transition-colors`}>
                <Icon className="h-4 w-4" />
              </div>
              <div className="ml-3">
                <p className="text-sm font-medium text-neutral-900 group-hover:text-neutral-700">
                  {action.title}
                </p>
                <p className="text-xs text-neutral-500">
                  {action.description}
                </p>
              </div>
            </Link>
          )
        })}
      </div>
    </div>
  )
}

export default QuickActions