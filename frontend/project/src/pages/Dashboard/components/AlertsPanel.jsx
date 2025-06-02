function AlertsPanel({ alerts }) {
  // Get icon and color based on alert type
  const getAlertInfo = (type) => {
    switch (type) {
      case 'warning':
        return {
          icon: (
            <svg className="w-5 h-5 text-warning-500" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
            </svg>
          ),
          bgColor: 'bg-warning-50',
          borderColor: 'border-warning-200',
        }
      case 'error':
        return {
          icon: (
            <svg className="w-5 h-5 text-error-500" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M10 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2m7-2a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          ),
          bgColor: 'bg-error-50',
          borderColor: 'border-error-200',
        }
      case 'success':
        return {
          icon: (
            <svg className="w-5 h-5 text-success-500" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          ),
          bgColor: 'bg-success-50',
          borderColor: 'border-success-200',
        }
      case 'info':
      default:
        return {
          icon: (
            <svg className="w-5 h-5 text-primary-500" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          ),
          bgColor: 'bg-primary-50',
          borderColor: 'border-primary-200',
        }
    }
  }
  
  return (
    <div className="space-y-3">
      {alerts.length === 0 ? (
        <div className="text-center py-4 text-neutral-500">No alerts at the moment</div>
      ) : (
        alerts.map(alert => {
          const { icon, bgColor, borderColor } = getAlertInfo(alert.type)
          
          return (
            <div
              key={alert.id}
              className={`p-3 rounded-md border ${borderColor} ${bgColor} flex items-start animate-fade-in`}
            >
              <div className="mr-3 mt-0.5">
                {icon}
              </div>
              <div className="flex-1">
                <p className="text-neutral-800">{alert.message}</p>
                <p className="text-xs text-neutral-500 mt-1">{alert.timestamp}</p>
              </div>
              <button className="text-neutral-400 hover:text-neutral-600">
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>
          )
        })
      )}
      
      {alerts.length > 0 && (
        <div className="text-center pt-2">
          <button className="text-sm text-primary-600 hover:text-primary-700 font-medium">
            View all alerts
          </button>
        </div>
      )}
    </div>
  )
}

export default AlertsPanel