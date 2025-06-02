import { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { supabase } from '../../services/supabase'
import Button from '../../components/common/Button'
import Input from '../../components/common/Input'
import Card from '../../components/common/Card'

function Signup() {
  const navigate = useNavigate()
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [formData, setFormData] = useState({
    email: '',
    password: '',
    confirmPassword: ''
  })

  const handleChange = (e) => {
    const { name, value } = e.target
    setFormData(prev => ({ ...prev, [name]: value }))
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    setLoading(true)
    setError(null)

    if (formData.password !== formData.confirmPassword) {
      setError("Passwords don't match")
      setLoading(false)
      return
    }

    try {
      const { error } = await supabase.auth.signUp({
        email: formData.email,
        password: formData.password
      })

      if (error) throw error

      navigate('/login')
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
          <h1 className="text-2xl font-bold text-neutral-900">Create Account</h1>
          <p className="text-neutral-600 mt-2">Sign up for a new account</p>
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
            helperText="Must be at least 8 characters"
          />

          <Input
            label="Confirm Password"
            type="password"
            id="confirmPassword"
            name="confirmPassword"
            value={formData.confirmPassword}
            onChange={handleChange}
            required
          />

          {error && (
            <div className="text-sm text-error-600 mt-2">{error}</div>
          )}

          <Button
            type="submit"
            className="w-full"
            isLoading={loading}
          >
            Create Account
          </Button>
        </form>

        <div className="mt-6 text-center text-sm">
          <span className="text-neutral-600">Already have an account? </span>
          <Link 
            to="/login" 
            className="text-primary-600 hover:text-primary-700 font-medium"
          >
            Sign in
          </Link>
        </div>
      </Card>
    </div>
  )
}

export default Signup