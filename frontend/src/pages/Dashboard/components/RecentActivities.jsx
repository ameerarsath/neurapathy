import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  Activity, 
  Brain, 
  Stethoscope, 
  Bell, 
  User, 
  Settings, 
  Calendar,
  Clock,
  ChevronRight,
  Filter,
  Eye
} from 'lucide-react'
import Card from '@components/common/Card'
import Button from '@components/common/Button'

function RecentActivities({ activities }) {
  const [filter, setFilter] = useState('all')
  const [showAll, setShowAll] = useState(false)

  if (!activities || activities.length === 0) {
    return (
      <Card title="Recent Activities" className="h-full">
        <div className="flex flex-col items-center justify-center h-40 text-gray-500 dark:text-gray-400">
          <Activity className="w-8 h-8 mb-2" />
          <p className="text-sm">No recent activities</p>
        </div>
      </Card>
    )
  }

  const getActivityIcon = (type) => {
    switch (type) {
      case 'test': return Stethoscope
      case 'prediction': return Brain
      case 'device': return Activity
      case 'alert': return Bell
      case 'user': return User
      case 'settings': return Settings
      default: return Activity
    }
  }

  const getActivityColor = (color) => {
    switch (color) {
      case 'blue': return 'text-blue-600 bg-blue-100 dark:bg-blue-900/20'
      case 'green': return 'text-green-600 bg-green-100 dark:bg-green-900/20'
      case 'yellow': return 'text-yellow-600 bg-yellow-100 dark:bg-yellow-900/20'
      case 'red': return 'text-red-600 bg-red-100 dark:bg-red-900/20'
      case 'purple': return 'text-purple-600 bg-purple-100 dark:bg-purple-900/20'
      default: return 'text-gray-600 bg-gray-100 dark:bg-gray-900/20'
    }
  }

  const formatTimeAgo = (timestamp) => {
    const now = new Date()
    const time = new Date(timestamp)
    const diffInMinutes = Math.floor((now - time) / (1000 * 60))

    if (diffInMinutes < 1) return 'Just now'
    if (diffInMinutes < 60) return `${diffInMinutes}m ago`
    if (diffInMinutes < 1440) return `${Math.floor(diffInMinutes / 60)}h ago`
    return `${Math.floor(diffInMinutes / 1440)}d ago`
  }

  const filteredActivities = activities.filter(activity => 
    filter === 'all' || activity.type === filter
  )

  const displayActivities = showAll ? filteredActivities : filteredActivities.slice(0, 5)

  const filterOptions = [
    { value: 'all', label: 'All', count: activities.length },
    { value: 'test', label: 'Tests', count: activities.filter(a => a.type === 'test').length },
    { value: 'prediction', label: 'Predictions', count: activities.filter(a => a.type === 'prediction').length },
    { value: 'device', label: 'Device', count: activities.filter(a => a.type === 'device').length },
    { value: 'alert', label: 'Alerts', count: activities.filter(a => a.type === 'alert').length }
  ]

  return (
    <Card 
      title="Recent Activities" 
      className="h-full"
      actions={
        <div className="flex items-center gap-2">
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setShowAll(!showAll)}
          >
            <Eye className="w-4 h-4" />
          </Button>
        </div>
      }
    >
      <div className="space-y-4">
        {/* Filter Tabs */}
        <div className="flex flex-wrap gap-1">
          {filterOptions.map((option) => (
            <button
              key={option.value}
              onClick={() => setFilter(option.value)}
              className={`px-2 py-1 text-xs font-medium rounded-md transition-colors ${
                filter === option.value
                  ? 'bg-blue-100 text-blue-700 dark:bg-blue-900/20 dark:text-blue-300'
                  : 'text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white hover:bg-gray-100 dark:hover:bg-gray-700'
              }`}
            >
              {option.label}
              {option.count > 0 && (
                <span className="ml-1 text-xs opacity-70">({option.count})</span>
              )}
            </button>
          ))}
        </div>

        {/* Activities List */}
        <div className="space-y-2">
          <AnimatePresence>
            {displayActivities.map((activity, index) => {
              const Icon = getActivityIcon(activity.type)
              
              return (
                <motion.div
                  key={activity.id}
                  initial={{ opacity: 0, x: -10 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: -10 }}
                  transition={{ delay: index * 0.05 }}
                  className="flex items-start gap-3 p-2 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-800/50 transition-colors cursor-pointer group"
                >
                  {/* Activity Icon */}
                  <div className={`p-1.5 rounded-lg ${getActivityColor(activity.color)}`}>
                    <Icon className="w-3 h-3" />
                  </div>

                  {/* Activity Content */}
                  <div className="flex-1 min-w-0">
                    <div className="flex items-start justify-between">
                      <div className="flex-1 min-w-0">
                        <p className="text-sm font-medium text-gray-900 dark:text-white truncate">
                          {activity.title}
                        </p>
                        <p className="text-xs text-gray-600 dark:text-gray-400 mt-0.5 line-clamp-2">
                          {activity.description}
                        </p>
                      </div>
                      <ChevronRight className="w-3 h-3 text-gray-400 ml-2 opacity-0 group-hover:opacity-100 transition-opacity" />
                    </div>
                    
                    {/* Timestamp */}
                    <div className="flex items-center gap-1 mt-1">
                      <Clock className="w-3 h-3 text-gray-400" />
                      <span className="text-xs text-gray-500 dark:text-gray-400">
                        {formatTimeAgo(activity.timestamp)}
                      </span>
                    </div>
                  </div>
                </motion.div>
              )
            })}
          </AnimatePresence>
        </div>

        {/* Show More/Less Button */}
        {filteredActivities.length > 5 && (
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setShowAll(!showAll)}
            className="w-full text-xs"
          >
            {showAll ? 'Show Less' : `Show All (${filteredActivities.length - 5} more)`}
          </Button>
        )}

        {/* No Activities Message */}
        {filteredActivities.length === 0 && filter !== 'all' && (
          <div className="text-center py-4">
            <p className="text-sm text-gray-500 dark:text-gray-400">
              No {filter} activities found
            </p>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setFilter('all')}
              className="mt-2 text-xs"
            >
              Show all activities
            </Button>
          </div>
        )}

        {/* Activity Summary */}
        <div className="pt-3 border-t border-gray-200 dark:border-gray-700">
          <div className="grid grid-cols-2 gap-2 text-xs text-center">
            <div>
              <div className="font-medium text-blue-600">
                {activities.filter(a => a.type === 'test').length}
              </div>
              <div className="text-gray-500">Tests Today</div>
            </div>
            <div>
              <div className="font-medium text-green-600">
                {activities.filter(a => a.type === 'prediction').length}
              </div>
              <div className="text-gray-500">Predictions</div>
            </div>
          </div>
        </div>

        {/* Quick Action */}
        <Button variant="outline" size="sm" className="w-full">
          <Calendar className="w-4 h-4 mr-2" />
          View Full Timeline
        </Button>
      </div>
    </Card>
  )
}

export default RecentActivities