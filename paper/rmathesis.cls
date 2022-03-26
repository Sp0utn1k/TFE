\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{rmathesis}
    [2020/12/14 v1.0 Standard RMA thesis class]

\newif\if@french    \@frenchfalse
\newif\if@dutch     \@dutchfalse
\newif\if@english   \@englishfalse

\DeclareOption{french}{%
	\@frenchtrue
}
\DeclareOption{dutch}{%
	\@dutchtrue
}
\DeclareOption{english}{%
	\@englishtrue
}

\PassOptionsToClass{11pt}{scrbook}
\ProcessOptions\relax
\LoadClass{scrbook}

\RequirePackage{polyglossia}
\if@french
	\newcommand{\trilingual}[3]{\if !#1! #3 \else #1 \fi}
	\setdefaultlanguage{french}
\else
	\if@dutch
		\newcommand{\trilingual}[3]{\if !#2! #3 \else #2 \fi}
		\setdefaultlanguage{dutch}
	\else
		\newcommand{\trilingual}[3]{\if !#3! #3 \else #3 \fi}
		\setdefaultlanguage{english}
	\fi
\fi

\newcommand\years[1]{\gdef\@years{#1}}
\def\@years{\@latex@warning@no@line{No \noexpand\years given}}

\newcommand\rank[1]{\gdef\@rank{#1}}
\def\@rank{\@latex@warning@no@line{No \noexpand\rank given}}

\newcommand\promotion[1]{\gdef\@promotion{#1}}
\def\@promotion{\@latex@warning@no@line{No \noexpand\promotion given}}

\newcommand\godfather[1]{\gdef\@godfather{#1}}
\def\@godfather{\@latex@warning@no@line{No \noexpand\godfather given}}

\newcommand\department[1]{\gdef\@department{#1}}
\def\@department{\@latex@warning@no@line{No \noexpand\department given}}

\newcommand\promotor[1]{\gdef\@promotor{#1}}
\def\@promotor{\@latex@warning@no@line{No \noexpand\promotor given}}

\newcommand\version[1]{\gdef\@version{#1}}
\def\@version{\@latex@warning@no@line{No \noexpand\version given}}

\RequirePackage{graphicx}

\RequirePackage{amsmath}
\RequirePackage{amsmath}
\RequirePackage{amsfonts}
\RequirePackage{bm}


\RequirePackage{libertinus}
\RequirePackage[condensed]{cabin}
\RequirePackage[scale=0.8]{FiraMono} 
\unimathsetup{math-style=ISO,bold-style=ISO,partial=upright}

\RequirePackage[a4paper]{geometry}
\geometry{verbose,tmargin=2cm,bmargin=2.5cm,lmargin=2.5cm,rmargin=2.5cm,footskip=1cm}

\RequirePackage{scrlayer-scrpage}

\setlength{\parskip}{\smallskipamount}
\setlength{\parindent}{0pt}

\KOMAoptions{toc=listof}
\KOMAoptions{bibliography=toc}

\setkomafont{pageheadfoot}{\sffamily\bfseries\footnotesize}
\setkomafont{pagenumber}{\sffamily\bfseries\footnotesize}

%\KOMAoptions{footsepline=true,plainfootsepline=true}
\ihead[]{}
\chead[]{}
\ohead[]{}
\ifoot[]{}
\cfoot[]{}
\rofoot[\pagemark]{\pagemark}
\lofoot[]{}
\lefoot[]{\pagemark}
\refoot[]{}
\pagestyle{scrheadings}
\automark[section]{chapter}

\RequirePackage[usenames,dvipsnames]{xcolor}

\renewcommand{\raggedchapter}{\centering}
\renewcommand*{\chapterformat}{\makebox[0.5cm]{\thechapter\autodot}}
\renewcommand*{\chapterlinesformat}[3]{
	\colorbox{lightgray}{
		\parbox{\dimexpr\linewidth-2\fboxsep}{
			\raggedchapter
			#2 \ #3
		}
	}
}

\KOMAoptions{captions=oneline}
\setkomafont{caption}{\small}
\setkomafont{captionlabel}{\sffamily\bfseries\footnotesize}
\renewcommand*{\captionformat}{\ }

\RequirePackage[intoc]{nomencl}
\makenomenclature

\bibliographystyle{alpha}

\addto\captionsenglish{%
	\renewcommand{\refname}{References}%
	\renewcommand{\nomname}{List of Abbreviations}%
}
\addto\captionsdutch{%
	\renewcommand{\bibname}{Bibliografie}%
	\renewcommand{\nomname}{Lijst met afkortingen}%
	\renewcommand{\listfigurename}{Lijst met figuren}%
	\renewcommand{\listtablename}{Lijst met tabellen}%
}
\addto\captionsfrench{%
	\renewcommand{\bibname}{Bibliographie}%
	\renewcommand{\nomname}{Liste des abréviations}%
	\renewcommand{\listfigurename}{Liste des graphiques et figures}%
	\renewcommand{\listtablename}{Liste des tableaux}%
}

\RequirePackage{listings}
\lstdefinelanguage{Julia}%
{morekeywords={abstract,break,case,catch,const,continue,do,else,elseif,%
		end,export,false,for,function,mutable,import,importall,if,in,%
		macro,module,otherwise,quote,return,struct,switch,true,try,type,typealias,%
		using,while},%
	sensitive=true,%
	alsoother={$},%
	morecomment=[l]\#,%
	morecomment=[n]{\#=}{=\#},%
	morestring=[s]{"}{"},%
	morestring=[m]{'}{'},%
}[keywords,comments,strings]%

\lstset{
	basicstyle       = \ttfamily\small,
	keywordstyle     = \bfseries\color{blue},
	stringstyle      = \color{magenta},
	commentstyle     = \color{ForestGreen},
	showstringspaces = false,
	tabsize          = 4,
}

\renewcommand\maketitle{
	\begin{titlepage}
		\begin{minipage}{\linewidth}
			\begin{minipage}{7cm}
				\begin{center}
					\large
					\trilingual{\underline{\textbf{ECOLE ROYALE MILITAIRE}}}{\underline{\textbf{KONINKLIJKE MILITAIRE SCHOOL}}}{\underline{\textbf{ROYAL MILITARY ACADEMY}}}\\\vspace{2mm}
					\@promotion\ \trilingual{Promotion POL}{Promotie POL}{Promotion POL}\\\vspace{2mm}
					\@godfather
				\end{center}
			\end{minipage}
			\hspace*{\stretch{1}}
			\begin{minipage}{7cm}
				\begin{center}
					\large
					\textbf{\trilingual{Année académique}{Academiejaar}{Academic year}\@years}\\\vspace{2mm}
					\trilingual{2$^{\text{\scriptsize ième}}$}{2$^{\text{\scriptsize de}}$}{2$^{\text{\scriptsize nd}}$}Master
				\end{center}
			\end{minipage}
		\end{minipage}
		\vspace*{\stretch{1}}
		\\
		\begin{minipage}{\linewidth}
			\begin{center}
				\textbf{\sffamily\huge\@title\\\vspace{1cm}
				\LARGE\@subtitle}
			\end{center}
		\end{minipage}
		\vspace*{\stretch{1}}
		\\
		\begin{minipage}{\linewidth}
			\begin{center}
				\large
				\trilingual{Par}{}{}\@rank\ \trilingual{candidat officier de carrière}{kandidaat-beroepsofficier}{Officer Cadet}\\\vspace{2mm}
				\@author
			\end{center}
		\end{minipage}
		\vspace*{\stretch{1}}
		\\
		\begin{minipage}{\linewidth}
			\begin{minipage}{3cm}
					\includegraphics[width=3cm]{images/logo-rma}
			\end{minipage}
			\hspace*{\stretch{1}}
			\begin{minipage}{13cm}
				\raggedleft
				\trilingual{Mémoire de fin d’études du département}{Masterproef van het departement}{Master Thesis of the department}\@department\\
				\trilingual{présenté en vue de l’obtention du grade académique}{voorgelegd tot het behalen van de academische graad}{presented to obtain the academic degree}\\
				\trilingual{de master en sciences de l'ingénieur}{van master in de ingenieurswetenschappen}{of Master in Engineering Science}\\
				\trilingual{sous la direction}{onder leiding van}{under the supervision of}\@promotor\\
				\trilingual{Bruxelles,}{Brussel,}{Brussels,}\the\year\
			\end{minipage}
		\end{minipage}
	\end{titlepage}
	\begin{titlepage}
		\begin{minipage}{\linewidth}
			\begin{minipage}{8cm}
				\begin{center}
					\large
					\trilingual{\underline{\textbf{ECOLE ROYALE MILITAIRE}}}{\underline{\textbf{KONINKLIJKE MILITAIRE SCHOOL}}}{\underline{\textbf{ROYAL MILITARY ACADEMY}}}\\\vspace{2mm}
					\trilingual{Département}{Departement}{Department}\@department
				\end{center}
			\end{minipage}
			\hspace*{\stretch{1}}
			\begin{minipage}{3cm}
				\begin{center}
					\large
					\textbf{\trilingual{Version}{Versie}{Version}\@version}\\\vspace{2mm}
					\the\day/\the\month/\the\year
				\end{center}
			\end{minipage}
		\end{minipage}
		\vspace*{\stretch{1}}
		\\
		\begin{minipage}{\linewidth}
			\begin{center}
				\textbf{\sffamily\huge\@title\\\vspace{1cm}
				\LARGE\@subtitle}
			\end{center}
		\end{minipage}
		\vspace*{\stretch{1}}
		\\
		\begin{minipage}{\linewidth}
			\begin{center}
				\large
				\@author
			\end{center}
		\end{minipage}
		\vspace*{\stretch{1}}
		\\
		\begin{center}
			\includegraphics[width=3cm]{images/logo-rma}
		\end{center}
	\end{titlepage}
 	\restoregeometry
}