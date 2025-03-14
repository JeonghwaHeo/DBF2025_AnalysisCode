<h1 align="center">FORCAST <br><br>
<i>Team Korea - Flight Optimized RC Aircraft Sizing Tool</i></h1> 
<br>

`FORCAST` is a aircraft sizing tool for the RC fixed-wing aircraft developed by  
**Team Korea** (`Seoul National University`, `Gyeongsang National University`, `Konkuk University`) for the [AIAA DBF 2025 competition](https://www.aiaa.org/dbf).
<p align="center">
<img src="https://github.com/user-attachments/assets/c2cbf36e-4ca1-4522-a76a-e7c6e58d1c29" alt="DBF Logo" width="300" height="100" hspace="50">  <img src="https://github.com/user-attachments/assets/127f68c1-2b93-4be0-b578-9a29a297234c" alt="DBF Logo" width="100" height="100" hspace="50">
</p>

## ⭐ Usage

It can be effectively used in the following situations.
- When finding the optimal aircraft design parameters for a predetermined flight path
- When predicting flight time, remaining battery capacity, and other performance metrics with a fixed aircraft configuration and flight path
- When obtaining aerodynamic analysis results for a given configuration
<br></br>

Although the code is structured for the AIAA DBF 2025 mission, it can be easily modified to suit individual objectives. Detailed information on the code structure is provided on the following website.



## 🔎 About Our Algorithm

Overall algorithm of FORCAST is presented below.
<br>
<p align="center">
<img src="https://github.com/user-attachments/assets/002db087-425f-4680-830d-34d3b8d3fb9d" alt="Sizing Tool Flowchart" width="650" height="550" >
</p>
<br></br>

## 📋 Used API

<p align="center">
  <img src="https://github.com/user-attachments/assets/0db09e04-e6df-412b-806e-b5120beb89ac" alt="OpenVSP Logo" width="400" height="100"> </p>
<p align="center">  
  <a href="https://openvsp.org/pyapi_docs/latest/">OpenVSP Python API</a>
</p>
<br></br>

## 📝 User Manual 

Supports on `Window + Docker Environment` , `Linux(Ubuntu)` , `Multiple Server Environment` .  

Aerodynamic analysis for a single design configuration takes approximately **15 seconds**, while performance analysis for a single flight configuration takes around **0.68 seconds**. If the grid search range is extensive, it is recommended to run the process in a multiple server environment.
<br></br>

### ❔ How to use

**[Instructions for Window + Docker](https://imaginary-spatula-125.notion.site/Instructions-for-Window-Docker-195bc109e8c4802f81dfc81afa3d7d09)**
<br>
**[Instructions for Linux + High Performance Server](https://imaginary-spatula-125.notion.site/Instructions-for-Linux-High-Performance-Server-195bc109e8c480efa8cdcb3af39ccc06)**
<br></br>

## 👪 Contributors

| Position    | Name        | University                            | Contact                    |
|-------------|-------------|---------------------------------------|----------------------------|
| Team Leader | [Jaewoo Lee](https://github.com/jwleesnu) | Seoul National University             | jaewoolee930@snu.ac.kr     |
| Developer (Aerodynamic Analysis)   | [Sungjin Kim](https://github.com/SungJinKm)      | Seoul National University        | ksjsms@snu.ac.kr            |
| Developer (Flight Analysis)  | [Gyeongrak Choe](https://github.com/Gyeong-Rak)      | Seoul National University                     | rudfkr5978@snu.ac.kr         |
| Developer (Code Structure)    |  [Jaewon Chung](https://github.com/PresidentPlant)      | Seoul National University             | jaewonchung7@snu.ac.kr         |
| Developer (Score Analysis)    |  [Hyeonjung Na](https://github.com/efq3)     | Gyeongsang National University             | nahj183@gnu.ac.kr         |
| Researcher (Propulsion Analysis)    |  Jeongheum Baek     | Seoul National University             | jamesb0103@snu.ac.kr       |
| Researcher (Propulsion Analysis)    |  [Sungil Son](https://github.com/One-star11)     | Seoul National University             | aldridge99@snu.ac.kr      |
| Researcher (Aerodynamic Analysis)    |  Seongwon Jang     | Seoul National University             | jsw0616wwn@snu.ac.kr      |
| Researcher (Aerodynamic Analysis)    |  Hyunchang Kweon     | Seoul National University             | thomas426789@gmail.com      |
| Researcher (Flight Analysis)    |  [Jeonghwa Heo](https://github.com/JeonghwaHeo)     | Seoul National University             | hjhcnn23@snu.ac.kr   |
| Researcher (Flight Analysis)    |  Junyeong Koo     | Seoul National University             | junyeongkoo@snu.ac.kr    |
| Researcher (Flight Analysis)    |  Seunghoon Ha     | Seoul National University             |   kerosene1@snu.ac.kr |
| Researcher (Flight Analysis)    |  Seojin Kim     | Gyeongsang National University            |   jingseo2@gnu.ac.kr |
| Researcher (Flight Analysis)    |  Junyoung Lee     | Seoul National University             |   99andrewlee@snu.ac.kr |
