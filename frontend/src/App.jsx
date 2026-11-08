import { Routes, Route, useLocation } from 'react-router-dom'
import { AnimatePresence } from 'framer-motion'
import Layout from './components/Layout'
import Landing from './pages/Landing'
import Concepts from './pages/Concepts'
import Simulator from './pages/Simulator'
import Detector from './pages/Detector'
import Results from './pages/Results'
import RLHonesty from './pages/RLHonesty'
import Playground from './pages/Playground'
import About from './pages/About'

export default function App() {
  const location = useLocation()
  return (
    <Layout>
      <AnimatePresence mode="wait">
        <Routes location={location} key={location.pathname}>
          <Route path="/" element={<Landing />} />
          <Route path="/concepts" element={<Concepts />} />
          <Route path="/simulator" element={<Simulator />} />
          <Route path="/detector" element={<Detector />} />
          <Route path="/results" element={<Results />} />
          <Route path="/rl-honesty" element={<RLHonesty />} />
          <Route path="/playground" element={<Playground />} />
          <Route path="/about" element={<About />} />
        </Routes>
      </AnimatePresence>
    </Layout>
  )
}
