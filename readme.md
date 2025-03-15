# SP100-Index-Tracker

An AI-driven index fund construction project that tracks the S&P 100 using a reduced set of stocks while maintaining similar performance.

## Table of Contents / İçindekiler Tablosu / Tabla de Contenidos

- [English](#english)
  - [Project Overview](#project-overview)
  - [Project Structure](#project-structure)
  - [Setup Instructions](#setup-instructions)
  - [Usage Instructions](#usage-instructions)
  - [AMPL Requirements](#ampl-requirements)
  - [Dependencies](#dependencies)
  - [Authors](#authors)
  - [License](#license)
  - [Acknowledgments](#acknowledgments)
- [Türkçe](#türkçe)
  - [Proje Genel Bakış](#proje-genel-bakış)
  - [Proje Yapısı](#proje-yapısı)
  - [Kurulum Talimatları](#kurulum-talimatları)
  - [Kullanım Talimatları](#kullanım-talimatları)
  - [AMPL Gereksinimleri](#ampl-gereksinimleri)
  - [Bağımlılıklar](#bağımlılıklar)
  - [Yazarlar](#yazarlar)
  - [Lisans](#lisans)
  - [Teşekkürler](#teşekkürler)
- [Español](#español)
  - [Descripción General del Proyecto](#descripción-general-del-proyecto)
  - [Estructura del Proyecto](#estructura-del-proyecto)
  - [Instrucciones de Instalación](#instrucciones-de-instalación)
  - [Instrucciones de Uso](#instrucciones-de-uso)
  - [Requisitos de AMPL](#requisitos-de-ampl)
  - [Dependencias](#dependencias)
  - [Autores](#autores)
  - [Licencia](#licencia)
  - [Agradecimientos](#agradecimientos)

---

# English

## Project Overview

This project uses optimization and machine learning techniques to create an index fund that tracks the S&P 100 with fewer stocks. It compares two approaches:
1. **Mathematical Optimization using AMPL**: Formulates the index tracking problem as a mathematical optimization and solves it using AMPL.
2. **Clustering & Genetic Algorithm**: An alternative approach using stock clustering and genetic algorithms to create optimized portfolios.

The project evaluates how well each approach tracks the S&P 100 index across different time periods and with different numbers of stocks (q).

## Project Structure

```
SP100-Index-Tracker/
├── README.md                       # Project overview and instructions
├── requirements.txt                # Python dependencies
├── data/                           # Data directory (created during execution)
├── results/                        # Results directory (created during execution)
│   ├── ampl/                       # AMPL results
│   ├── ga/                         # Genetic Algorithm results
│   └── visualizations/             # Generated charts and figures
├── src/                            # Source code
│   ├── 1_data_collection.py        # Collects S&P 100 data
│   ├── 2_ampl_data_preparation.py  # Prepares data for AMPL
│   ├── 2a_run_ampl_optimization.py # Runs AMPL optimization via Python API
│   ├── 3_alternative_approach.py   # Clustering and GA implementation
│   ├── 4_visualization_analysis.py # Analysis and visualization
│   └── utils/                      # Utility functions
│       ├── __init__.py
│       └── helpers.py              # Helper functions
└── ampl/                           # AMPL model files
    ├── index_fund.mod              # AMPL model file
    ├── index_fund.run              # AMPL run commands
    └── README.md                   # AMPL setup instructions
```

## Setup Instructions

### Prerequisites
- Python 3.8 or later
- AMPL with a suitable solver (e.g., CPLEX, Gurobi)
- Git (for cloning the repository)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/SP100-Index-Tracker.git
cd SP100-Index-Tracker
```

2. Create and activate a virtual environment (optional but recommended):
```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up AMPL:
   - Install AMPL following the instructions at [ampl.com](https://ampl.com/products/ampl/ampl-for-students/)
   - Place your AMPL license key in the appropriate location:
     - For macOS/Linux: Save the license key in `~/.ampl/`
     - For Windows: Save the license key in `C:\Users\YourUsername\.ampl\`
   - Make sure the AMPL executable is in your system PATH
   - For detailed instructions, see the `ampl/README.md` file

## Usage Instructions

Follow these steps to run the project:

1. **Data Collection**:
```bash
python src/1_data_collection.py
```
This step downloads S&P 100 index data and performs necessary preparations.

2. **AMPL Data Preparation**:
```bash
python src/2_ampl_data_preparation.py
```
This step prepares the data files needed for AMPL optimization.

3. **Run AMPL Optimization**:
```bash
python src/2a_run_ampl_optimization.py
```
This step runs AMPL optimization via the Python API.

4. **Run Alternative Approach (Clustering & Genetic Algorithm)**:
```bash
python src/3_alternative_approach.py
```
This step performs portfolio optimization using clustering and genetic algorithms.

5. **Visualization and Analysis**:
```bash
python src/4_visualization_analysis.py
```
This step analyzes and visualizes the results of the different approaches.

**Note**: The visualization script can find result files in different locations. If files are in a different location from a previous step, it automatically finds and analyzes them. This feature ensures the project works smoothly on different computers and operating systems.

## AMPL Requirements

This project requires AMPL with a mathematical solver such as CPLEX or Gurobi. The script `2a_run_ampl_optimization.py` uses the `amplpy` Python API to interact with AMPL.

### AMPL License Key Setup
1. Obtain a license key from AMPL (student or commercial)
2. Place the license file in the correct location:
   - macOS/Linux: `~/.ampl/license.lic`
   - Windows: `C:\Users\YourUsername\.ampl\license.lic`
3. Alternative: Set the environment variable `AMPLKEY` with your license key:
   ```bash
   # macOS/Linux
   export AMPLKEY="your-license-key"
   
   # Windows (PowerShell)
   $env:AMPLKEY="your-license-key"
   ```

For more detailed instructions, see the AMPL documentation at [ampl.com/resources](https://ampl.com/resources/).

## Dependencies

- numpy - For numerical operations
- pandas - For data manipulation
- matplotlib - For plotting
- seaborn - For enhanced visualizations
- scikit-learn - For clustering and machine learning
- deap - For genetic algorithm implementation
- yfinance - For downloading financial data
- amplpy - For interfacing with AMPL (only needed for step 3)

See `requirements.txt` for specific versions.

## Authors

- [Dr. Sam]
- [Team Member Name]

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- National College of Ireland
- H9AIDM: AI Driven Decision Making course
- Instructors: Ade Fajemisin, Harshani Nagahamulla

---

# Türkçe

## Proje Genel Bakış

Bu proje, S&P 100'ü daha az hisse senedi kullanarak takip eden bir endeks fonu oluşturmak için optimizasyon ve makine öğrenimi teknikleri kullanır. İki yaklaşımı karşılaştırır:
1. **AMPL Kullanarak Matematiksel Optimizasyon**: Endeks takip problemini matematiksel optimizasyon olarak formüle eder ve AMPL kullanarak çözer.
2. **Kümeleme ve Genetik Algoritma**: Optimize edilmiş portföyler oluşturmak için hisse senedi kümelemesi ve genetik algoritmaları kullanan alternatif bir yaklaşım.

Proje, her bir yaklaşımın farklı zaman periyotlarında ve farklı sayıda hisse senedi (q) ile S&P 100 endeksini ne kadar iyi takip ettiğini değerlendirir.

## Proje Yapısı

```
SP100-Index-Tracker/
├── README.md                       # Proje genel bakış ve talimatlar
├── requirements.txt                # Python bağımlılıkları
├── data/                           # Veri dizini (çalıştırma sırasında oluşturulur)
├── results/                        # Sonuçlar dizini (çalıştırma sırasında oluşturulur)
│   ├── ampl/                       # AMPL sonuçları
│   ├── ga/                         # Genetik Algoritma sonuçları
│   └── visualizations/             # Oluşturulan grafikler ve şekiller
├── src/                            # Kaynak kodu
│   ├── 1_data_collection.py        # S&P 100 verilerini toplar
│   ├── 2_ampl_data_preparation.py  # AMPL için veri hazırlar
│   ├── 2a_run_ampl_optimization.py # Python API ile AMPL optimizasyonunu çalıştırır
│   ├── 3_alternative_approach.py   # Kümeleme ve GA uygulaması
│   ├── 4_visualization_analysis.py # Analiz ve görselleştirme
│   └── utils/                      # Yardımcı fonksiyonlar
│       ├── __init__.py
│       └── helpers.py              # Yardımcı fonksiyonlar
└── ampl/                           # AMPL model dosyaları
    ├── index_fund.mod              # AMPL model dosyası
    ├── index_fund.run              # AMPL çalıştırma komutları
    └── README.md                   # AMPL kurulum talimatları
```

## Kurulum Talimatları

### Önkoşullar
- Python 3.8 veya üstü
- Uygun bir çözücü ile AMPL (örn. CPLEX, Gurobi)
- Git (depoyu klonlamak için)

### Kurulum

1. Depoyu klonlayın:
```bash
git clone https://github.com/your-username/SP100-Index-Tracker.git
cd SP100-Index-Tracker
```

2. Sanal bir ortam oluşturun ve etkinleştirin (isteğe bağlı ama önerilir):
```bash
# Windows için
python -m venv venv
venv\Scripts\activate

# macOS/Linux için
python -m venv venv
source venv/bin/activate
```

3. Bağımlılıkları yükleyin:
```bash
pip install -r requirements.txt
```

4. AMPL'ı kurun:
   - [ampl.com](https://ampl.com/products/ampl/ampl-for-students/) adresindeki talimatları izleyerek AMPL'ı yükleyin
   - AMPL lisans anahtarınızı uygun konuma yerleştirin:
     - macOS/Linux için: Lisans anahtarını `~/.ampl/` konumuna kaydedin
     - Windows için: Lisans anahtarını `C:\Users\KullanıcıAdınız\.ampl\` konumuna kaydedin
   - AMPL çalıştırılabilir dosyasının sistem PATH'inizde olduğundan emin olun
   - Ayrıntılı talimatlar için `ampl/README.md` dosyasına bakın

## Kullanım Talimatları

Projeyi çalıştırmak için aşağıdaki adımları izleyin:

1. **Veri Toplama**:
```bash
python src/1_data_collection.py
```
Bu adım, S&P 100 endeks verilerini indirir ve gerekli hazırlıkları yapar.

2. **AMPL Verilerini Hazırlama**:
```bash
python src/2_ampl_data_preparation.py
```
Bu adım, AMPL optimizasyonu için gereken veri dosyalarını hazırlar.

3. **AMPL Optimizasyonunu Çalıştırma**:
```bash
python src/2a_run_ampl_optimization.py
```
Bu adım, Python API aracılığıyla AMPL optimizasyonunu çalıştırır.

4. **Alternatif Yaklaşımı Çalıştırma (Kümeleme ve Genetik Algoritma)**:
```bash
python src/3_alternative_approach.py
```
Bu adım, kümeleme ve genetik algoritma kullanarak portföy optimizasyonu gerçekleştirir.

5. **Görselleştirme ve Analiz**:
```bash
python src/4_visualization_analysis.py
```
Bu adım, farklı yaklaşımların sonuçlarını analiz eder ve görselleştirir.

**Not**: Görselleştirme betiği, sonuç dosyalarını farklı konumlarda arayabilir. Dosyalar önceki bir adımdan farklı bir konumdaysa, onları otomatik olarak bulur ve analiz eder. Bu özellik, projenin farklı bilgisayarlarda ve işletim sistemlerinde sorunsuz çalışmasını sağlar.

## AMPL Gereksinimleri

Bu proje, CPLEX veya Gurobi gibi bir matematiksel çözücü içeren AMPL'ı gerektirir. `2a_run_ampl_optimization.py` betiği, AMPL ile etkileşim kurmak için `amplpy` Python API'sini kullanır.

### AMPL Lisans Anahtarı Kurulumu
1. AMPL'dan bir lisans anahtarı edinin (öğrenci veya ticari)
2. Lisans dosyasını doğru konuma yerleştirin:
   - macOS/Linux: `~/.ampl/license.lic`
   - Windows: `C:\Users\KullanıcıAdınız\.ampl\license.lic`
3. Alternatif: `AMPLKEY` çevre değişkenini lisans anahtarınızla ayarlayın:
   ```bash
   # macOS/Linux
   export AMPLKEY="lisans-anahtarınız"
   
   # Windows (PowerShell)
   $env:AMPLKEY="lisans-anahtarınız"
   ```

Daha ayrıntılı talimatlar için [ampl.com/resources](https://ampl.com/resources/) adresindeki AMPL belgelerine bakın.

## Bağımlılıklar

- numpy - Sayısal işlemler için
- pandas - Veri manipülasyonu için
- matplotlib - Çizim için
- seaborn - Gelişmiş görselleştirmeler için
- scikit-learn - Kümeleme ve makine öğrenimi için
- deap - Genetik algoritma uygulaması için
- yfinance - Finansal verileri indirmek için
- amplpy - AMPL ile arayüz oluşturmak için (sadece 3. adım için gerekli)

Belirli sürümler için `requirements.txt` dosyasına bakın.

## Yazarlar

- [Dr. Sam]
- [Team Member Name]

## Lisans

Bu proje MIT Lisansı altında lisanslanmıştır - ayrıntılar için LICENSE dosyasına bakın.

## Teşekkürler

- National College of Ireland
- H9AIDM: AI Driven Decision Making dersi
- Eğitmenler: Ade Fajemisin, Harshani Nagahamulla

---

# Español

## Descripción General del Proyecto

Este proyecto utiliza técnicas de optimización y aprendizaje automático para crear un fondo índice que sigue el S&P 100 utilizando menos acciones. Compara dos enfoques:
1. **Optimización Matemática usando AMPL**: Formula el problema de seguimiento del índice como una optimización matemática y lo resuelve usando AMPL.
2. **Agrupación y Algoritmo Genético**: Un enfoque alternativo utilizando agrupación de acciones y algoritmos genéticos para crear carteras optimizadas.

El proyecto evalúa qué tan bien cada enfoque sigue el índice S&P 100 a través de diferentes períodos de tiempo y con diferentes números de acciones (q).

## Estructura del Proyecto

```
SP100-Index-Tracker/
├── README.md                       # Descripción general e instrucciones del proyecto
├── requirements.txt                # Dependencias de Python
├── data/                           # Directorio de datos (creado durante la ejecución)
├── results/                        # Directorio de resultados (creado durante la ejecución)
│   ├── ampl/                       # Resultados de AMPL
│   ├── ga/                         # Resultados del Algoritmo Genético
│   └── visualizations/             # Gráficos y figuras generados
├── src/                            # Código fuente
│   ├── 1_data_collection.py        # Recopila datos del S&P 100
│   ├── 2_ampl_data_preparation.py  # Prepara datos para AMPL
│   ├── 2a_run_ampl_optimization.py # Ejecuta optimización AMPL vía API de Python
│   ├── 3_alternative_approach.py   # Implementación de agrupación y GA
│   ├── 4_visualization_analysis.py # Análisis y visualización
│   └── utils/                      # Funciones de utilidad
│       ├── __init__.py
│       └── helpers.py              # Funciones auxiliares
└── ampl/                           # Archivos de modelo AMPL
    ├── index_fund.mod              # Archivo de modelo AMPL
    ├── index_fund.run              # Comandos de ejecución AMPL
    └── README.md                   # Instrucciones de configuración AMPL
```

## Instrucciones de Instalación

### Prerrequisitos
- Python 3.8 o posterior
- AMPL con un solucionador adecuado (por ejemplo, CPLEX, Gurobi)
- Git (para clonar el repositorio)

### Instalación

1. Clonar el repositorio:
```bash
git clone https://github.com/your-username/SP100-Index-Tracker.git
cd SP100-Index-Tracker
```

2. Crear y activar un entorno virtual (opcional pero recomendado):
```bash
# En Windows
python -m venv venv
venv\Scripts\activate

# En macOS/Linux
python -m venv venv
source venv/bin/activate
```

3. Instalar dependencias:
```bash
pip install -r requirements.txt
```

4. Configurar AMPL:
   - Instalar AMPL siguiendo las instrucciones en [ampl.com](https://ampl.com/products/ampl/ampl-for-students/)
   - Colocar su clave de licencia AMPL en la ubicación apropiada:
     - Para macOS/Linux: Guardar la clave de licencia en `~/.ampl/`
     - Para Windows: Guardar la clave de licencia en `C:\Users\SuNombreDeUsuario\.ampl\`
   - Asegurarse de que el ejecutable AMPL esté en su PATH del sistema
   - Para instrucciones detalladas, ver el archivo `ampl/README.md`

## Instrucciones de Uso

Siga estos pasos para ejecutar el proyecto:

1. **Recopilación de Datos**:
```bash
python src/1_data_collection.py
```
Este paso descarga los datos del índice S&P 100 y realiza las preparaciones necesarias.

2. **Preparación de Datos para AMPL**:
```bash
python src/2_ampl_data_preparation.py
```
Este paso prepara los archivos de datos necesarios para la optimización AMPL.

3. **Ejecutar Optimización AMPL**:
```bash
python src/2a_run_ampl_optimization.py
```
Este paso ejecuta la optimización AMPL a través de la API de Python.

4. **Ejecutar Enfoque Alternativo (Agrupación y Algoritmo Genético)**:
```bash
python src/3_alternative_approach.py
```
Este paso realiza la optimización de cartera utilizando agrupación y algoritmos genéticos.

5. **Visualización y Análisis**:
```bash
python src/4_visualization_analysis.py
```
Este paso analiza y visualiza los resultados de los diferentes enfoques.

**Nota**: El script de visualización puede encontrar archivos de resultados en diferentes ubicaciones. Si los archivos están en una ubicación diferente de un paso anterior, los encuentra y analiza automáticamente. Esta característica asegura que el proyecto funcione sin problemas en diferentes computadoras y sistemas operativos.

## Requisitos de AMPL

Este proyecto requiere AMPL con un solucionador matemático como CPLEX o Gurobi. El script `2a_run_ampl_optimization.py` utiliza la API de Python `amplpy` para interactuar con AMPL.

### Configuración de la Clave de Licencia AMPL
1. Obtener una clave de licencia de AMPL (estudiante o comercial)
2. Colocar el archivo de licencia en la ubicación correcta:
   - macOS/Linux: `~/.ampl/license.lic`
   - Windows: `C:\Users\SuNombreDeUsuario\.ampl\license.lic`
3. Alternativa: Establecer la variable de entorno `AMPLKEY` con su clave de licencia:
   ```bash
   # macOS/Linux
   export AMPLKEY="su-clave-de-licencia"
   
   # Windows (PowerShell)
   $env:AMPLKEY="su-clave-de-licencia"
   ```

Para instrucciones más detalladas, consulte la documentación de AMPL en [ampl.com/resources](https://ampl.com/resources/).

## Dependencias

- numpy - Para operaciones numéricas
- pandas - Para manipulación de datos
- matplotlib - Para gráficos
- seaborn - Para visualizaciones mejoradas
- scikit-learn - Para agrupación y aprendizaje automático
- deap - Para implementación de algoritmos genéticos
- yfinance - Para descargar datos financieros
- amplpy - Para interfaz con AMPL (solo necesario para el paso 3)

Ver `requirements.txt` para versiones específicas.

## Autores

- [Dr. Sam]
- [Team Member Name]

## Licencia

Este proyecto está licenciado bajo la Licencia MIT - vea el archivo LICENSE para más detalles.

## Agradecimientos

- National College of Ireland
- H9AIDM: Curso de Toma de Decisiones Impulsada por IA
- Instructores: Ade Fajemisin, Harshani Nagahamulla
