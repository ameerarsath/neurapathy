function SensitivityHeatmap() {
  // This component would use a more sophisticated visualization in a real app
  
  return (
    <div className="space-y-6">
      <div className="flex flex-wrap gap-4 justify-center">
        {/* Left foot heatmap */}
        <div className="text-center">
          <h3 className="text-sm font-medium mb-2">Left Foot Sensitivity</h3>
          <div className="w-[180px] h-[300px] relative bg-neutral-100 rounded-lg overflow-hidden">
            {/* Foot outline */}
            <svg width="180" height="300" viewBox="0 0 180 300" xmlns="http://www.w3.org/2000/svg">
              <path 
                d="M50,50 C30,100 20,150 20,200 C20,250 30,300 60,350 C90,400 150,400 180,350 C210,300 220,250 220,200 C220,150 210,100 190,50 C170,0 70,0 50,50 Z" 
                fill="none" 
                stroke="#d1d5db" 
                strokeWidth="1" 
                transform="scale(0.8) translate(10, -20)"
              />
            </svg>
            
            {/* Heatmap overlay */}
            <div className="absolute inset-0 opacity-70">
              <div className="absolute left-[20%] top-[20%] w-[40%] h-[25%] bg-gradient-radial from-green-400 to-transparent rounded-full transform -translate-x-1/2 -translate-y-1/2"></div>
              <div className="absolute left-[70%] top-[25%] w-[35%] h-[20%] bg-gradient-radial from-green-500 to-transparent rounded-full transform -translate-x-1/2 -translate-y-1/2"></div>
              <div className="absolute left-[20%] top-[50%] w-[50%] h-[30%] bg-gradient-radial from-yellow-400 to-transparent rounded-full transform -translate-x-1/2 -translate-y-1/2"></div>
              <div className="absolute left-[70%] top-[50%] w-[45%] h-[25%] bg-gradient-radial from-yellow-500 to-transparent rounded-full transform -translate-x-1/2 -translate-y-1/2"></div>
              <div className="absolute left-[45%] top-[75%] w-[60%] h-[40%] bg-gradient-radial from-orange-500 to-transparent rounded-full transform -translate-x-1/2 -translate-y-1/2"></div>
            </div>
          </div>
        </div>
        
        {/* Right foot heatmap */}
        <div className="text-center">
          <h3 className="text-sm font-medium mb-2">Right Foot Sensitivity</h3>
          <div className="w-[180px] h-[300px] relative bg-neutral-100 rounded-lg overflow-hidden">
            {/* Foot outline */}
            <svg width="180" height="300" viewBox="0 0 180 300" xmlns="http://www.w3.org/2000/svg">
              <path 
                d="M50,50 C30,100 20,150 20,200 C20,250 30,300 60,350 C90,400 150,400 180,350 C210,300 220,250 220,200 C220,150 210,100 190,50 C170,0 70,0 50,50 Z" 
                fill="none" 
                stroke="#d1d5db" 
                strokeWidth="1" 
                transform="scale(0.8) translate(10, -20)"
              />
            </svg>
            
            {/* Heatmap overlay */}
            <div className="absolute inset-0 opacity-70">
              <div className="absolute left-[20%] top-[20%] w-[40%] h-[25%] bg-gradient-radial from-yellow-400 to-transparent rounded-full transform -translate-x-1/2 -translate-y-1/2"></div>
              <div className="absolute left-[70%] top-[25%] w-[35%] h-[20%] bg-gradient-radial from-yellow-500 to-transparent rounded-full transform -translate-x-1/2 -translate-y-1/2"></div>
              <div className="absolute left-[20%] top-[50%] w-[50%] h-[30%] bg-gradient-radial from-orange-400 to-transparent rounded-full transform -translate-x-1/2 -translate-y-1/2"></div>
              <div className="absolute left-[70%] top-[50%] w-[45%] h-[25%] bg-gradient-radial from-orange-500 to-transparent rounded-full transform -translate-x-1/2 -translate-y-1/2"></div>
              <div className="absolute left-[45%] top-[75%] w-[60%] h-[40%] bg-gradient-radial from-red-500 to-transparent rounded-full transform -translate-x-1/2 -translate-y-1/2"></div>
            </div>
          </div>
        </div>
      </div>
      
      {/* Legend */}
      <div className="max-w-md mx-auto">
        <div className="flex justify-center mb-2">
          <div className="flex items-center space-x-1">
            <span className="text-xs text-neutral-600">Normal</span>
            <div className="w-full h-4 bg-gradient-to-r from-green-500 via-yellow-500 to-red-500 rounded"></div>
            <span className="text-xs text-neutral-600">No Sensitivity</span>
          </div>
        </div>
        
        <div className="space-y-4 mt-6">
          <div className="border-t border-neutral-200 pt-4">
            <h4 className="text-sm font-medium mb-2">Key Observations</h4>
            <ul className="list-disc pl-5 space-y-1 text-sm text-neutral-700">
              <li>Right foot shows significantly reduced sensitivity compared to left foot</li>
              <li>Both feet show pronounced sensitivity loss in the heel areas</li>
              <li>The right foot has developed a concerning area of minimal sensitivity at the ball of the foot</li>
              <li>Toe sensitivity is relatively preserved in the left foot but diminished in the right foot</li>
              <li>Progression pattern suggests peripheral neuropathy typical of diabetic complication</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  )
}

export default SensitivityHeatmap