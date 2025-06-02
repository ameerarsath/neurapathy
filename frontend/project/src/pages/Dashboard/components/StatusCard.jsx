function StatusCard({ label, value, icon, change, changeType, info }) {
  // Render appropriate icon based on icon name
  const renderIcon = () => {
    switch (icon) {
      case 'chart':
        return (
          <svg className="w-6 h-6 text-primary-500" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M7 12l3-3 3 3 4-4M8 21l4-4 4 4M3 4h18M4 4h16v12a1 1 0 01-1 1H5a1 1 0 01-1-1V4z" />
          </svg>
        )
      case 'check':
        return (
          <svg className="w-6 h-6 text-success-500" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
        )
      case 'foot':
        return (
          <svg className="w-6 h-6 text-secondary-500" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" />
          </svg>
        )
      case 'calendar':
        return (
          <svg className="w-6 h-6 text-accent-500" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
          </svg>
        )
      default:
        return null
    }
  }
  
  // Get color for change indicator
  const getChangeColor = () => {
    switch (changeType) {
      case 'positive':
        return 'text-success-500'
      case 'negative':
        return 'text-error-500'
      case 'neutral':
      default:
        return 'text-neutral-500'
    }
  }
  
  return (
    <div className="bg-white p-4 rounded-lg border border-neutral-200 shadow-card">
      <div className="flex justify-between items-start">
        <div>
          <p className="text-sm font-medium text-neutral-500">{label}</p>
          <p className="mt-1 text-xl font-semibold text-neutral-900">{value}</p>
        </div>
        <div className="p-2 bg-neutral-100 rounded-md">
          {renderIcon()}
        </div>
      </div>
      
      {/* Change indicator */}
      {change && (
        <div className="mt-2 flex items-center">
          {changeType === 'positive' && (
            <svg className="w-4 h-4 text-success-500" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 10l7-7m0 0l7 7m-7-7v18" />
            </svg>
          )}
          {changeType === 'negative' && (
            <svg className="w-4 h-4 text-error-500" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 14l-7 7m0 0l-7-7m7 7V3" />
            </svg>
          )}
          <span className={`text-sm ml-1 ${getChangeColor()}`}>{change}</span>
        </div>
      )}
      
      {/* Info tooltip */}
      {info && (
        <div className="mt-2 text-xs text-neutral-500">{info}</div>
      )}
    </div>
  )
}

export default StatusCard