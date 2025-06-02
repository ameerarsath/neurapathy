function ProgressionChart() {
  // This is a placeholder component for the progression chart
  // In a real app, you would use a charting library like Chart.js or Recharts
  
  return (
    <div className="h-[300px] relative">
      {/* Chart visualization */}
      <svg className="w-full h-full" viewBox="0 0 400 200" preserveAspectRatio="none">
        {/* Grid lines */}
        <line x1="50" y1="20" x2="50" y2="180" stroke="#e5e7eb" strokeWidth="1" />
        <line x1="125" y1="20" x2="125" y2="180" stroke="#e5e7eb" strokeWidth="1" />
        <line x1="200" y1="20" x2="200" y2="180" stroke="#e5e7eb" strokeWidth="1" />
        <line x1="275" y1="20" x2="275" y2="180" stroke="#e5e7eb" strokeWidth="1" />
        <line x1="350" y1="20" x2="350" y2="180" stroke="#e5e7eb" strokeWidth="1" />
        
        <line x1="50" y1="20" x2="350" y2="20" stroke="#e5e7eb" strokeWidth="1" />
        <line x1="50" y1="60" x2="350" y2="60" stroke="#e5e7eb" strokeWidth="1" />
        <line x1="50" y1="100" x2="350" y2="100" stroke="#e5e7eb" strokeWidth="1" />
        <line x1="50" y1="140" x2="350" y2="140" stroke="#e5e7eb" strokeWidth="1" />
        <line x1="50" y1="180" x2="350" y2="180" stroke="#e5e7eb" strokeWidth="1" />
        
        {/* Left foot line */}
        <path
          d="M50,60 C80,62 110,70 140,80 C170,90 200,100 230,115 C260,130 290,140 320,145 C330,147 340,148 350,150"
          fill="none"
          stroke="#0077CC"
          strokeWidth="2"
        />
        
        {/* Right foot line */}
        <path
          d="M50,60 C80,65 110,75 140,90 C170,105 200,125 230,140 C260,155 290,165 320,170 C330,172 340,173 350,175"
          fill="none"
          stroke="#6941C6"
          strokeWidth="2"
        />
        
        {/* Data points - Left foot */}
        <circle cx="50" cy="60" r="4" fill="#0077CC" />
        <circle cx="125" cy="80" r="4" fill="#0077CC" />
        <circle cx="200" cy="100" r="4" fill="#0077CC" />
        <circle cx="275" cy="125" r="4" fill="#0077CC" />
        <circle cx="350" cy="150" r="4" fill="#0077CC" />
        
        {/* Data points - Right foot */}
        <circle cx="50" cy="60" r="4" fill="#6941C6" />
        <circle cx="125" cy="90" r="4" fill="#6941C6" />
        <circle cx="200" cy="125" r="4" fill="#6941C6" />
        <circle cx="275" cy="155" r="4" fill="#6941C6" />
        <circle cx="350" cy="175" r="4" fill="#6941C6" />
      </svg>
      
      {/* Y-axis labels */}
      <div className="absolute left-0 top-0 h-full flex flex-col justify-between text-xs text-neutral-500 py-5">
        <div>100%</div>
        <div>75%</div>
        <div>50%</div>
        <div>25%</div>
        <div>0%</div>
      </div>
      
      {/* X-axis labels */}
      <div className="absolute left-[50px] right-0 bottom-0 flex justify-between text-xs text-neutral-500">
        <div>Jan '23</div>
        <div>Apr '23</div>
        <div>Jul '23</div>
        <div>Oct '23</div>
        <div>Jan '24</div>
      </div>
      
      {/* Legend */}
      <div className="absolute top-2 right-4 flex items-center space-x-4 bg-white p-1 rounded border border-neutral-200 text-xs">
        <div className="flex items-center">
          <div className="w-3 h-3 rounded-full bg-primary-500 mr-1"></div>
          <span>Left Foot</span>
        </div>
        <div className="flex items-center">
          <div className="w-3 h-3 rounded-full bg-accent-500 mr-1"></div>
          <span>Right Foot</span>
        </div>
      </div>
    </div>
  )
}

export default ProgressionChart