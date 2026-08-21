import { useState } from 'react';
import './App.css';
import BrowseView from './BrowseView';
import PredictionView from './PredictionView';

/**
 * View controller. Two screens does not justify a router.
 *
 * scheduleGames lives here rather than in BrowseView so that going Back
 * from a prediction returns to the list already fetched, instead of
 * hitting the NBA schedule API again.
 */
function App() {
  const [view, setView] = useState('browse');
  const [scheduleGames, setScheduleGames] = useState([]);
  const [selectedResult, setSelectedResult] = useState(null);

  function handleSelect(result) {
    setSelectedResult(result);
    setView('detail');
  }

  return (
    <div className="app">
      {view === 'browse' ? (
        <BrowseView
          games={scheduleGames}
          onGamesLoaded={setScheduleGames}
          onSelect={handleSelect}
        />
      ) : (
        <PredictionView
          result={selectedResult}
          onBack={() => setView('browse')}
        />
      )}
    </div>
  );
}

export default App;
