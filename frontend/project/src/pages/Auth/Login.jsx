import { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { supabase } from '../../services/supabase'
import Button from '../../components/common/Button'
import Input from '../../components/common/Input'
import Card from '../../components/common/Card'

function Login() {
  const navigate = useNavigate()
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [formData, setFormData] = useState({
    email: '',
    password: ''
  })

  const handleChange = (e) => {
    const { name, value } = e.target
    setFormData(prev => ({ ...prev, [name]: value }))
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    setLoading(true)
    setError(null)

    try {
      const { error } = await supabase.auth.signInWithPassword({
        email: formData.email,
        password: formData.password
      })

      if (error) throw error

      navigate('/')
    } catch (error) {
      setError(error.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-neutral-50 flex items-center justify-center p-4">
      <Card className="w-full max-w-md">
        <div className="text-center mb-8">
          <h1 className="text-2xl font-bold text-neutral-900">Welcome Back</h1>
          <p className="text-neutral-600 mt-2">Sign in to your account</p>
        </div>

        <form onSubmit={handleSubmit} className="space-y-4">
          <Input
            label="Email Address"
            type="email"
            id="email"
            name="email"
            value={formData.email}
            onChange={handleChange}
            required
          />

          <Input
            label="Password"
            type="password"
            id="password"
            name="password"
            value={formData.password}
            onChange={handleChange}
            required
          />

          {error && (
            <div className="text-sm text-error-600 mt-2">{error}</div>
          )}

          <div className="flex items-center justify-between">
            <Link 
              to="/forgot-password" 
              className="text-sm text-primary-600 hover:text-primary-700"
            >
              Forgot your password?
            </Link>
          </div>

          <Button
            type="submit"
            className="w-full"
            isLoading={loading}
          >
            Sign In
          </Button>
        </form>

        <div className="mt-6 text-center text-sm">
          <span className="text-neutral-600">Don't have an account? </span>
          <Link 
            to="/signup" 
            className="text-primary-600 hover:text-primary-700 font-medium"
          >
            Sign up
          </Link>
        </div>
      </Card>
    </div>
  )
}

export default Login