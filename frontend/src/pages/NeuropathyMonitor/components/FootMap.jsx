function FootMap() {
  // In a real app, this would be an interactive visualization
  // Here we're just showing a static representation
  
  return (
    <div className="flex flex-col items-center">
      <div className="flex space-x-8 mb-4">
        {/* Left foot */}
        <div className="relative">
          <h3 className="text-center text-sm font-medium mb-2">Left Foot</h3>
          <svg width="120" height="240" viewBox="0 0 100 200" xmlns="http://www.w3.org/2000/svg">
            {/* Foot outline */}
            <path 
              d="M30,30 C20,50 15,80 15,100 C15,130 20,150 40,180 C60,210 80,210 90,180 C100,150 105,130 105,100 C105,80 100,50 90,30 C80,10 40,10 30,30 Z" 
              fill="#f3f4f6" 
              stroke="#d1d5db" 
              strokeWidth="1" 
            />
            
            {/* Sensitivity zones */}
            <circle cx="35" cy="50" r="10" fill="#6941C620" stroke="#6941C640" />
            <circle cx="70" cy="50" r="10" fill="#6941C640" stroke="#6941C660" />
            <circle cx="30" cy="100" r="12" fill="#6941C670" stroke="#6941C690" />
            <circle cx="70" cy="100" r="12" fill="#6941C650" stroke="#6941C670" />
            <circle cx="50" cy="150" r="15" fill="#6941C680" stroke="#6941C6A0" />
            
            {/* Zone labels */}
            <text x="35" y="50" textAnchor="middle" dominantBaseline="middle" fill="#374151" fontSize="8" fontWeight="bold">1</text>
            <text x="70" y="50" textAnchor="middle" dominantBaseline="middle" fill="#374151" fontSize="8" fontWeight="bold">2</text>
            <text x="30" y="100" textAnchor="middle" dominantBaseline="middle" fill="#374151" fontSize="8" fontWeight="bold">3</text>
            <text x="70" y="100" textAnchor="middle" dominantBaseline="middle" fill="#374151" fontSize="8" fontWeight="bold">4</text>
            <text x="50" y="150" textAnchor="middle" dominantBaseline="middle" fill="#374151" fontSize="8" fontWeight="bold">5</text>
          </svg>
        </div>
        
        {/* Right foot */}
        <div className="relative">
          <h3 className="text-center text-sm font-medium mb-2">Right Foot</h3>
          <svg width="120" height="240" viewBox="0 0 100 200" xmlns="http://www.w3.org/2000/svg">
            {/* Foot outline */}
            <path 
              d="M30,30 C20,50 15,80 15,100 C15,130 20,150 40,180 C60,210 80,210 90,180 C100,150 105,130 105,100 C105,80 100,50 90,30 C80,10 40,10 30,30 Z" 
              fill="#f3f4f6" 
              stroke="#d1d5db" 
              strokeWidth="1" 
            />
            
            {/* Sensitivity zones */}
            <circle cx="35" cy="50" r="10" fill="#6941C640" stroke="#6941C660" />
            <circle cx="70" cy="50" r="10" fill="#6941C660" stroke="#6941C680" />
            <circle cx="30" cy="100" r="12" fill="#6941C690" stroke="#6941C6B0" />
            <circle cx="70" cy="100" r="12" fill="#6941C6A0" stroke="#6941C6C0" />
            <circle cx="50" cy="150" r="15" fill="#6941C6B0" stroke="#6941C6D0" />
            
            {/* Zone labels */}
            <text x="35" y="50" textAnchor="middle" dominantBaseline="middle" fill="#374151" fontSize="8" fontWeight="bold">1</text>
            <text x="70" y="50" textAnchor="middle" dominantBaseline="middle" fill="#374151" fontSize="8" fontWeight="bold">2</text>
            <text x="30" y="100" textAnchor="middle" dominantBaseline="middle" fill="#374151" fontSize="8" fontWeight="bold">3</text>
            <text x="70" y="100" textAnchor="middle" dominantBaseline="middle" fill="#374151" fontSize="8" fontWeight="bold">4</text>
            <text x="50" y="150" textAnchor="middle" dominantBaseline="middle" fill="#374151" fontSize="8" fontWeight="bold">5</text>
          </svg>
        </div>
      </div>
      
      {/* Legend */}
      <div className="w-full max-w-xs">
        <div className="flex justify-center mb-2">
          <div className="flex items-center space-x-2">
            <span className="text-xs text-neutral-600">Low Sensitivity</span>
            <div className="w-24 h-4 bg-gradient-to-r from-accent-200 to-accent-900 rounded"></div>
            <span className="text-xs text-neutral-600">High Sensitivity</span>
          </div>
        </div>
        
        <div className="flex justify-between text-xs text-neutral-500">
          <div>Healthy</div>
          <div>Mild</div>
          <div>Moderate</div>
          <div>Severe</div>
        </div>
      </div>
      
      {/* Sensitivity readings table */}
      <div className="mt-6 w-full max-w-md">
        <h4 className="text-sm font-medium mb-2">Zone Sensitivity Readings</h4>
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-neutral-200">
            <thead>
              <tr>
                <th className="px-3 py-2 text-left text-xs font-medium text-neutral-500 uppercase tracking-wider">Zone</th>
                <th className="px-3 py-2 text-left text-xs font-medium text-neutral-500 uppercase tracking-wider">Left Foot</th>
                <th className="px-3 py-2 text-left text-xs font-medium text-neutral-500 uppercase tracking-wider">Right Foot</th>
                <th className="px-3 py-2 text-left text-xs font-medium text-neutral-500 uppercase tracking-wider">Change</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-neutral-200">
              <tr>
                <td className="px-3 py-2 text-sm font-medium">Zone 1</td>
                <td className="px-3 py-2 text-sm">85%</td>
                <td className="px-3 py-2 text-sm">75%</td>
                <td className="px-3 py-2 text-sm text-error-600">-3%</td>
              </tr>
              <tr>
                <td className="px-3 py-2 text-sm font-medium">Zone 2</td>
                <td className="px-3 py-2 text-sm">78%</td>
                <td className="px-3 py-2 text-sm">66%</td>
                <td className="px-3 py-2 text-sm text-error-600">-5%</td>
              </tr>
              <tr>
                <td className="px-3 py-2 text-sm font-medium">Zone 3</td>
                <td className="px-3 py-2 text-sm">62%</td>
                <td className="px-3 py-2 text-sm">45%</td>
                <td className="px-3 py-2 text-sm text-error-600">-8%</td>
              </tr>
              <tr>
                <td className="px-3 py-2 text-sm font-medium">Zone 4</td>
                <td className="px-3 py-2 text-sm">70%</td>
                <td className="px-3 py-2 text-sm">40%</td>
                <td className="px-3 py-2 text-sm text-error-600">-12%</td>
              </tr>
              <tr>
                <td className="px-3 py-2 text-sm font-medium">Zone 5</td>
                <td className="px-3 py-2 text-sm">55%</td>
                <td className="px-3 py-2 text-sm">32%</td>
                <td className="px-3 py-2 text-sm text-error-600">-10%</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}

export default FootMap