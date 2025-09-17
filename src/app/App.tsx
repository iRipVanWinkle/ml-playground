import MLLayout from './MLLayout';
import { ThemeProvider } from './components/theme';
import { Toaster } from './components/ui/sonner';
import { Footer } from './components/Footer';
import { Header } from './components/Header';

import './App.css';

function App() {
    return (
        <ThemeProvider>
            <Header />
            <main>
                <MLLayout />
            </main>
            <Footer />
            <Toaster position="top-right" expand richColors closeButton />
        </ThemeProvider>
    );
}

export default App;
