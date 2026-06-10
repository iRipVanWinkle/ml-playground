import { Toaster } from './shared/ui';
import { ThemeProvider } from './features/change-theme';
import { DesignProvider } from './features/switch-design';
import { DesignRoot } from './DesignRoot';
import { Header } from './widgets/header';
import { Footer } from './widgets/footer';

function App() {
    return (
        <DesignProvider>
            <ThemeProvider>
                <div className="App">
                    <Header />

                    <DesignRoot />

                    <Footer />
                </div>
                <Toaster position="top-right" expand closeButton richColors />
            </ThemeProvider>
        </DesignProvider>
    );
}

export default App;
