function SensorReadings({ timeRange }) {
  // This is a placeholder component for the sensor readings chart
  // In a real app, you would use a charting library like Chart.js or Recharts
  
  return (
    <div className="relative">
      {/* Chart placeholder */}
      <div className="h-[250px] w-full relative">
        <div className="absolute inset-0 bg-gradient-to-r from-primary-500 to-secondary-500 opacity-10 rounded"></div>
        
        {/* Y-axis labels */}
        <div className="absolute left-0 top-0 h-full flex flex-col justify-between text-xs text-neutral-500 py-2">
          <div>100</div>
          <div>75</div>
          <div>50</div>
          <div>25</div>
          <div>0</div>
        </div>
        
        {/* Placeholder line */}
        <div className="absolute inset-0 ml-8 mr-4 mt-4 mb-6">
          <svg className="w-full h-full" viewBox="0 0 400 200" preserveAspectRatio="none">
            <path
              d="M0,150 C50,120 100,180 150,130 C200,80 250,100 300,70 C350,40 400,60 400,40"
              fill="none"
              stroke="#0077CC"
              strokeWidth="2"
            />
            <path
              d="M0,170 C50,160 100,140 150,160 C200,180 250,140 300,130 C350,120 400,140 400,120"
              fill="none"
              stroke="#00A3A3"
              strokeWidth="2"
            />
          </svg>
          
          {/* Data points */}
          <div className="absolute left-[10%] top-[60%] w-2 h-2 rounded-full bg-primary-500"></div>
          <div className="absolute left-[25%] top-[35%] w-2 h-2 rounded-full bg-primary-500"></div>
          <div className="absolute left-[40%] top-[65%] w-2 h-2 rounded-full bg-primary-500"></div>
          <div className="absolute left-[55%] top-[40%] w-2 h-2 rounded-full bg-primary-500"></div>
          <div className="absolute left-[70%] top-[35%] w-2 h-2 rounded-full bg-primary-500"></div>
          <div className="absolute left-[85%] top-[20%] w-2 h-2 rounded-full bg-primary-500"></div>
          
          <div className="absolute left-[10%] top-[85%] w-2 h-2 rounded-full bg-secondary-500"></div>
          <div className="absolute left-[25%] top-[70%] w-2 h-2 rounded-full bg-secondary-500"></div>
          <div className="absolute left-[40%] top-[80%] w-2 h-2 rounded-full bg-secondary-500"></div>
          <div className="absolute left-[55%] top-[90%] w-2 h-2 rounded-full bg-secondary-500"></div>
          <div className="absolute left-[70%] top-[65%] w-2 h-2 rounded-full bg-secondary-500"></div>
          <div className="absolute left-[85%] top-[60%] w-2 h-2 rounded-full bg-secondary-500"></div>
        </div>
        
        {/* X-axis labels */}
        <div className="absolute left-8 right-4 bottom-0 flex justify-between text-xs text-neutral-500">
          {timeRange === 'day' && (
            <>
              <div>12 AM</div>
              <div>4 AM</div>
              <div>8 AM</div>
              <div>12 PM</div>
              <div>4 PM</div>
              <div>8 PM</div>
            </>
          )}
          
          {timeRange === 'week' && (
            <>
              <div>Mon</div>
              <div>Tue</div>
              <div>Wed</div>
              <div>Thu</div>
              <div>Fri</div>
              <div>Sat</div>
              <div>Sun</div>
            </>
          )}
          
          {timeRange === 'month' && (
            <>
              <div>Week 1</div>
              <div>Week 2</div>
              <div>Week 3</div>
              <div>Week 4</div>
            </>
          )}
        </div>
      </div>
      
      {/* Legend */}
      <div className="mt-4 flex items-center justify-center space-x-6">
        <div className="flex items-center">
          <div className="w-3 h-3 bg-primary-500 rounded-full mr-2"></div>
          <span className="text-sm text-neutral-600">Temperature</span>
        </div>
        <div className="flex items-center">
          <div className="w-3 h-3 bg-secondary-500 rounded-full mr-2"></div>
          <span className="text-sm text-neutral-600">Pressure</span>
        </div>
      </div>
      
      <div className="mt-4 text-center text-xs text-neutral-500">
        {timeRange === 'day' && 'Showing data for last 24 hours'}
        {timeRange === 'week' && 'Showing data for last 7 days'}
        {timeRange === 'month' && 'Showing data for last 30 days'}
      </div>
    </div>
  )
}

export default SensorReadings