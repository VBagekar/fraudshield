import React, { useState, useEffect } from 'react';
import api from './api/axios';

export default function App() {
  const [amount, setAmount] = useState('');
  const [merchantCategory, setMerchantCategory] = useState('Online');
  const [timeOfDay, setTimeOfDay] = useState('Morning');
  const [locationRiskScore, setLocationRiskScore] = useState('5');

  const [predictionResult, setPredictionResult] = useState(null);
  const [predictLoading, setPredictLoading] = useState(false);
  const [predictError, setPredictError] = useState(null);

  const [stats, setStats] = useState({
    totalTransactions: 0,
    fraudCount: 0,
    fraudRate: 0,
    avgProbability: 0
  });
  const [statsLoading, setStatsLoading] = useState(true);
  const [statsError, setStatsError] = useState(null);

  const [transactions, setTransactions] = useState([]);
  const [transactionsLoading, setTransactionsLoading] = useState(true);
  const [transactionsError, setTransactionsError] = useState(null);

  const fetchStats = async () => {
    try {
      setStatsError(null);
      const res = await api.get('/stats');
      const data = res.data;
      setStats({
        totalTransactions: data.total_transactions ?? data.totalTransactions ?? 0,
        fraudCount: data.fraud_count ?? data.fraudCount ?? 0,
        fraudRate: data.fraud_rate ?? data.fraudRate ?? 0,
        avgProbability: data.avg_fraud_probability ?? data.avgProbability ?? 0
      });
    } catch (err) {
      setStatsError(err.response?.data?.detail || err.message || 'Failed to fetch stats');
    } finally {
      setStatsLoading(false);
    }
  };

  const fetchTransactions = async () => {
    try {
      setTransactionsError(null);
      const res = await api.get('/transactions');
      const data = res.data;
      setTransactions(Array.isArray(data) ? data : (data.transactions || []));
    } catch (err) {
      setTransactionsError(err.response?.data?.detail || err.message || 'Failed to fetch transactions');
    } finally {
      setTransactionsLoading(false);
    }
  };

  useEffect(() => {
    fetchStats();
    fetchTransactions();
    const interval = setInterval(() => {
      fetchStats();
    }, 15000);
    return () => clearInterval(interval);
  }, []);

  const handlePredict = async (e) => {
    e.preventDefault();
    if (!amount) return;
    setPredictLoading(true);
    setPredictError(null);
    setPredictionResult(null);

    try {
      const res = await api.post('/predict', {
        amount: parseFloat(amount),
        merchant_category: merchantCategory,
        time_of_day: timeOfDay,
        location_risk_score: parseFloat(locationRiskScore)
      });
      setPredictionResult(res.data);
      // Refresh transactions and stats after a prediction
      fetchStats();
      fetchTransactions();
    } catch (err) {
      setPredictError(err.response?.data?.detail || err.message || 'Prediction request failed');
    } finally {
      setPredictLoading(false);
    }
  };

  const getRiskColor = (prob) => {
    if (prob > 0.7) return { border: 'border-red-500', bg: 'bg-red-950', text: 'text-red-500', bar: 'bg-red-500', label: 'HIGH RISK' };
    if (prob >= 0.3) return { border: 'border-amber-500', bg: 'bg-amber-950', text: 'text-amber-500', bar: 'bg-amber-500', label: 'SUSPICIOUS' };
    return { border: 'border-emerald-500', bg: 'bg-emerald-950', text: 'text-emerald-500', bar: 'bg-emerald-500', label: 'SAFE' };
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-200 font-sans">
      {/* Navbar */}
      <nav className="flex justify-between items-center px-6 py-4 bg-slate-900 border-b border-slate-800">
        <h1 className="text-2xl font-bold text-white tracking-tight">FraudShield</h1>
        <span className="text-sm font-medium text-slate-400">Live Detection System</span>
      </nav>

      {/* Main Layout */}
      <main className="flex flex-col lg:flex-row gap-6 p-6 max-w-[1600px] mx-auto">

        {/* LEFT COLUMN */}
        <div className="w-full lg:w-2/5 flex flex-col gap-6">
          <div className="bg-slate-800 rounded-xl border border-slate-700 p-6 shadow-xl">
            <h2 className="text-xl font-semibold text-white mb-6">Analyze Transaction</h2>

            <form onSubmit={handlePredict} className="flex flex-col gap-5">
              <div>
                <label className="block text-sm font-medium text-slate-400 mb-1">Amount (₹)</label>
                <input
                  type="number"
                  step="0.01"
                  min="0"
                  required
                  value={amount}
                  onChange={(e) => setAmount(e.target.value)}
                  className="w-full bg-slate-900 border border-slate-700 rounded-lg px-4 py-2 text-white focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-colors"
                  placeholder="0.00"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-slate-400 mb-1">Merchant Category</label>
                <select
                  value={merchantCategory}
                  onChange={(e) => setMerchantCategory(e.target.value)}
                  className="w-full bg-slate-900 border border-slate-700 rounded-lg px-4 py-2 text-white focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-colors appearance-none"
                >
                  <option value="Online">Online</option>
                  <option value="Retail">Retail</option>
                  <option value="Restaurant">Restaurant</option>
                  <option value="Travel">Travel</option>
                  <option value="Entertainment">Entertainment</option>
                  <option value="Gas">Gas</option>
                  <option value="Healthcare">Healthcare</option>
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-slate-400 mb-1">Time of Day</label>
                <select
                  value={timeOfDay}
                  onChange={(e) => setTimeOfDay(e.target.value)}
                  className="w-full bg-slate-900 border border-slate-700 rounded-lg px-4 py-2 text-white focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-colors appearance-none"
                >
                  <option value="Morning">Morning</option>
                  <option value="Afternoon">Afternoon</option>
                  <option value="Evening">Evening</option>
                  <option value="Night">Night</option>
                </select>
              </div>

              <div>
                <div className="flex justify-between items-center mb-1">
                  <label className="block text-sm font-medium text-slate-400">Location Risk Score</label>
                  <span className="text-sm font-bold text-blue-400">{locationRiskScore}/10</span>
                </div>
                <input
                  type="range"
                  min="0"
                  max="10"
                  step="0.1"
                  value={locationRiskScore}
                  onChange={(e) => setLocationRiskScore(e.target.value)}
                  className="w-full accent-blue-500"
                />
              </div>

              <button
                type="submit"
                disabled={predictLoading}
                className="w-full mt-2 bg-blue-600 hover:bg-blue-500 text-white font-semibold py-3 rounded-lg transition-colors flex justify-center items-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {predictLoading ? (
                  <span className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin"></span>
                ) : (
                  "Analyze Transaction"
                )}
              </button>

              {predictError && (
                <div className="text-red-400 text-sm mt-2 font-medium bg-red-950/30 p-3 rounded-lg border border-red-900/50">
                  {predictError}
                </div>
              )}
            </form>
          </div>

          {/* Result Card */}
          {predictionResult && !predictLoading && (
            <div className={`rounded-xl border p-6 flex flex-col gap-4 animate-in fade-in slide-in-from-bottom-4 duration-500 ${getRiskColor(predictionResult.fraud_probability).bg} ${getRiskColor(predictionResult.fraud_probability).border}`}>
              <div className="flex justify-between items-start">
                <div>
                  <h3 className="text-sm font-semibold text-slate-300 uppercase tracking-wider mb-1">Analysis Result</h3>
                  <div className={`px-2.5 py-1 inline-flex rounded-full text-[10px] font-bold border ${getRiskColor(predictionResult.fraud_probability).border} ${getRiskColor(predictionResult.fraud_probability).text} bg-slate-950/50 uppercase`}>
                    {predictionResult.risk_level || getRiskColor(predictionResult.fraud_probability).label}
                  </div>
                </div>
                <div className="text-right">
                  <div className={`text-5xl font-black ${getRiskColor(predictionResult.fraud_probability).text}`}>
                    {(predictionResult.fraud_probability * 100).toFixed(1)}%
                  </div>
                  <div className="text-xs font-medium text-slate-400 mt-1 uppercase">Fraud Probability</div>
                </div>
              </div>

              <div className="w-full bg-slate-900 rounded-full h-2 mt-2 overflow-hidden border border-slate-800">
                <div
                  className={`h-2 rounded-full transition-all duration-1000 ease-out ${getRiskColor(predictionResult.fraud_probability).bar}`}
                  style={{ width: `${predictionResult.fraud_probability * 100}%` }}
                ></div>
              </div>
            </div>
          )}
        </div>

        {/* RIGHT COLUMN */}
        <div className="w-full lg:w-3/5 flex flex-col gap-6">

          {/* STATS */}
          <div className="grid grid-cols-2 gap-4">
            <StatCard
              label="Total Analyzed"
              value={stats.totalTransactions?.toLocaleString() || '0'}
              loading={statsLoading}
              error={statsError}
            />
            <StatCard
              label="Fraud Detected"
              value={stats.fraudCount?.toLocaleString() || '0'}
              loading={statsLoading}
              error={statsError}
              valueColor="text-red-400"
            />
            <StatCard
              label="Fraud Rate"
              value={`${(stats.fraudRate || 0).toFixed(2)}%`}
              loading={statsLoading}
              error={statsError}
            />
            <StatCard
              label="Avg Risk"
              value={`${(stats.avgProbability * 100 || 0).toFixed(1)}%`}
              loading={statsLoading}
              error={statsError}
            />
          </div>

          {/* TABLE */}
          <div className="bg-slate-800 rounded-xl border border-slate-700 overflow-hidden shadow-xl flex-grow flex flex-col">
            <div className="p-6 border-b border-slate-700 flex justify-between items-center">
              <h2 className="text-xl font-semibold text-white">Recent Flagged Transactions</h2>
              {transactionsLoading && <span className="w-4 h-4 border-2 border-slate-400 border-t-transparent rounded-full animate-spin"></span>}
            </div>

            <div className="overflow-x-auto flex-grow">
              {transactionsError ? (
                <div className="p-6 text-red-400">{transactionsError}</div>
              ) : transactionsLoading && transactions.length === 0 ? (
                <div className="p-6 space-y-4">
                  {[1, 2, 3, 4, 5].map(i => (
                    <div key={i} className="h-10 bg-slate-700/50 rounded animate-pulse w-full"></div>
                  ))}
                </div>
              ) : (
                <table className="w-full text-left border-collapse min-w-max">
                  <thead>
                    <tr className="bg-slate-900/50 text-xs uppercase tracking-wider text-slate-400 border-b border-slate-700">
                      <th className="px-6 py-4 font-medium">Time</th>
                      <th className="px-6 py-4 font-medium">Amount</th>
                      <th className="px-6 py-4 font-medium">Category</th>
                      <th className="px-6 py-4 font-medium">Probability</th>
                      <th className="px-6 py-4 font-medium">Status</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-700/50 text-sm">
                    {transactions.map((tx, i) => {
                      const prob = tx.fraud_probability !== undefined ? tx.fraud_probability : tx.probability;
                      const formattedTime = tx.timestamp || tx.time || new Date().toISOString();
                      const amountStr = tx.amount !== undefined ? `₹${parseFloat(tx.amount).toFixed(2)}` : '₹0.00';
                      const { border, text, label } = getRiskColor(prob || 0);

                      let displayTime = '';
                      try {
                        const d = new Date(formattedTime);
                        const hh = String(d.getHours()).padStart(2, '0');
                        const mm = String(d.getMinutes()).padStart(2, '0');
                        const dd = String(d.getDate()).padStart(2, '0');
                        const mo = String(d.getMonth() + 1).padStart(2, '0');
                        displayTime = `${hh}:${mm} ${dd}/${mo}`;
                      } catch {
                        displayTime = formattedTime;
                      }

                      return (
                        <tr key={tx.id || tx.transaction_id || i} className="hover:bg-slate-700/30 transition-colors">
                          <td className="px-6 py-4 text-slate-300 tabular-nums whitespace-nowrap">
                            {displayTime}
                          </td>
                          <td className="px-6 py-4 font-medium text-white tabular-nums">{amountStr}</td>
                          <td className="px-6 py-4 text-slate-300">{tx.merchant_category || tx.category || 'N/A'}</td>
                          <td className="px-6 py-4 tabular-nums">
                            <span className={text}>{((prob || 0) * 100).toFixed(1)}%</span>
                          </td>
                          <td className="px-6 py-4">
                            <span className={`px-2 py-1 rounded text-[10px] font-bold border ${border} ${text} uppercase tracking-wider bg-slate-900/50`}>
                              {tx.risk_level || label}
                            </span>
                          </td>
                        </tr>
                      );
                    })}
                    {transactions.length === 0 && !transactionsLoading && (
                      <tr>
                        <td colSpan="5" className="px-6 py-12 text-center text-slate-500">
                          No flagged transactions found.
                        </td>
                      </tr>
                    )}
                  </tbody>
                </table>
              )}
            </div>
          </div>

        </div>
      </main>
    </div>
  );
}

function StatCard({ label, value, loading, error, valueColor = "text-white" }) {
  return (
    <div className="bg-slate-800 rounded-xl border border-slate-700 p-6 flex flex-col justify-center shadow-lg relative overflow-hidden">
      <div className="text-sm font-medium text-slate-400 mb-2 relative z-10">{label}</div>
      {error ? (
        <div className="text-xs text-red-400 mt-1 relative z-10">Error loading</div>
      ) : loading && (value === '0' || value === '0%' || value === '0.0%') ? (
        <div className="h-9 bg-slate-700/50 rounded w-24 animate-pulse relative z-10"></div>
      ) : (
        <div className={`text-4xl font-bold tracking-tight ${valueColor} relative z-10`}>
          {value}
        </div>
      )}
      {/* Background glow effect for premium feel */}
      <div className="absolute -right-6 -bottom-6 w-32 h-32 bg-slate-700/10 rounded-full blur-2xl z-0 pointer-events-none"></div>
      <div className="absolute -left-6 -top-6 w-24 h-24 bg-slate-700/5 rounded-full blur-xl z-0 pointer-events-none"></div>
    </div>
  );
}
