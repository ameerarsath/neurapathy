import React from 'react'
import { useQuery } from '@tanstack/react-query'
import { smartShoeAPI } from '../../services/api'
import { useAuth } from '../../contexts/AuthContext'
import LoadingSpinner from '../common/LoadingSpinner'
import { 
  Activity, 
  User, 
  Smartphone, 
  AlertTriangle,
  Clock,
  TrendingUp
} from 'lucide-react'
import { formatDistanceToNow } from 'date-fns'

const RecentActivity = () => {
  const { canAccess } = useAuth()

  // Fetch recent activity from API
  const { data: recentActivities = [] } = useQuery({
    queryKey: ['recent-activity'],
    queryFn: () => smartShoeAPI.dashboard.getRecentActivity(),
    select: data => data?.data || [],
    staleTime: 5 * 60 * 1000
  })

  const getColorClasses = (color) => {
    const classes = {
      primary: 'text-primary-600 bg-primary-50',
      secondary: 'text-secondary-600 bg-secondary-50',
      success: 'text-green-600 bg-green-50',
      warning: 'text-yellow-600 bg-yellow-50',
      error: 'text-red-600 bg-red-50',
      neutral: 'text-neutral-600 bg-neutral-50'
    }
    return classes[color] || classes.neutral
  }

  return (
    <div className="medical-card">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-medium text-neutral-900">
          Recent Activity
        </h3>
        <button className="text-sm text-primary-600 hover:text-primary-700">
          View all
        </button>
      </div>

      <div className="flow-root">
        <ul className="-mb-8">
          {Array.isArray(recentActivities) ? recentActivities.map((activity, activityIdx) => {
            const Icon = activity.icon
            return (
              <li key={activity.id}>
                <div className="relative pb-8">
                  {activityIdx !== recentActivities.length - 1 ? (
                    <span
                      className="absolute top-4 left-4 -ml-px h-full w-0.5 bg-neutral-200"
                      aria-hidden="true"
                    />
                  ) : null}
                  <div className="relative flex space-x-3">
                    <div>
                      <span className={`h-8 w-8 rounded-full flex items-center justify-center ring-8 ring-white ${getColorClasses(activity.color)}`}>
                        <Icon className="h-4 w-4" />
                      </span>
                    </div>
                    <div className="min-w-0 flex-1 pt-1.5 flex justify-between space-x-4">
                      <div>
                        <p className="text-sm font-medium text-neutral-900">
                          {activity.title}
                        </p>
                        <p className="text-sm text-neutral-500">
                          {activity.description}
                        </p>
                      </div>
                      <div className="text-right text-sm whitespace-nowrap text-neutral-500">
                        <div className="flex items-center">
                          <Clock className="h-3 w-3 mr-1" />
                          <time>{formatDistanceToNow(activity.timestamp, { addSuffix: true })}</time>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </li>
            )
          }) : (
            <li className="text-center py-8 text-gray-500">
              No recent activities
            </li>
          )}
        </ul>
      </div>

      {/* Real-time indicator */}
      <div className="mt-4 pt-4 border-t border-neutral-200">
        <div className="flex items-center text-xs text-neutral-500">
          <div className="h-2 w-2 bg-success rounded-full mr-2 animate-pulse"></div>
          Live updates enabled
        </div>
      </div>
    </div>
  )
}

export default RecentActivity