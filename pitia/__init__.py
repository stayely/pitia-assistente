"""
   Projeto de Assistente Virtual Avançado, Pítia.
   Copyright (C) 2025 Stayely
   Licenciado sob GPLv3 (https://www.gnu.org/licenses/gpl-3.0.txt)
   Autora: Stayely
"""
import requests
from bs4 import BeautifulSoup
from googlesearch import search
import re
import random
import pickle
import os
import nltk
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse
import time
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ssl
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import certifi
from collections import defaultdict
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
import subprocess

# baixa recursos necessários do nltk para tratamento de texto
nltk.download(['punkt', 'stopwords', 'punkt_tab', 'rslp'], quiet=True)
os.system("sudo dmesg -n 1")

class AssistenteAvancado:
    def __init__(self):
        # configuração de processamento de texto
        self.palavras_parada = set(stopwords.words('portuguese') + stopwords.words('english'))
        
        # inicialização de sistemas de conhecimentos
        self.base_conhecimento = defaultdict(list)
        self.respostas_aprendidas = {}
        self.carregar_conhecimento()
        
        # configurações de pesquisa, prioriza sites confiaveis 3-alto...
        self.dominios_confiaveis = {
            'wikipedia.org': 2,
            'gov.br': 3,
            'bbc.com': 2,
            'nationalgeographic.com': 2,
            'uol.com.br': 1,
            'terra.com.br': 1,
            'edu.br': 3
        }
        
        # configuração robusta da sessão de HTTP paraevitar instabilidade
        self.sessao = requests.Session()
        self.tentativas = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=[500, 502, 503, 504]
        )
        self.adaptador = HTTPAdapter(max_retries=self.tentativas)
        self.sessao.mount('http://', self.adaptador)
        self.sessao.mount('https://', self.adaptador)
        
        # configura SSL com certificados confiáveis
        self.sessao.verify = True
        self.sessao.cert = None
        self.contexto_ssl = ssl.create_default_context(cafile=certifi.where())
        self.contexto_ssl.minimum_version = ssl.TLSVersion.TLSv1_2
        
        # configuração de cabeçalhos para evitar bloqueios
        self.sessao.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept-Language': 'pt-BR,pt;q=0.9'
        })
        
        # processamento de texto(resume e onde paraleliza as tarefas)
        self.resumidor = LsaSummarizer()
        self.executor = ThreadPoolExecutor(max_workers=5)
        self.cache = {}
        self.mapa_simplificacao = {
            r"\bposteriormente\b": "depois",
            r"\bdevido ao fato de\b": "porque",
            r"\bconstitui\b": "é",
            r"\bdenominado\b": "chamado"
        }
        self.arquivo_memoria = "assistant_memory.json"
        self.respostas_aprendidas = self._carregar_memoria()
        self.vectorizador = TfidfVectorizer() # converte texto em vetores para comparação
        self._inicializar_vectorizador()
        self.ultima_pergunta = None
        self.ultima_resposta = None
        self.stemmer = nltk.stem.RSLPStemmer()
        self.ultima_consulta_curta = None  
        
        # variação de sinônimos para melhorar as respostas
        self.sinonimos = {
            'deus': ['divindade', 'entidade divina', 'ser supremo'],
            'mitologia': ['lendas antigas', 'histórias tradicionais', 'narrativas sagradas'],
            'grego': ['da Grécia', 'helênico', 'grego antigo'],
            'importante': ['relevante', 'significativo', 'crucial'],
            'pessoa': ['indivíduo', 'ser humano', 'cidadão']
        }
        
    def salvar_conhecimento(self):
        """salva a base de conhecimento em arquivo pickle(transforma em binário)"""
        try:
            with open('knowledge.pkl', 'wb') as f:
                pickle.dump(dict(self.base_conhecimento), f)
            return True
        except Exception as e:
            print(f"Erro ao salvar conhecimento: {e}")
            return False
    
    def carregar_conhecimento(self):
        """carrega a base de conhecimento do arquivo de antes"""
        try:
            if os.path.exists('knowledge.pkl'):
                with open('knowledge.pkl', 'rb') as f:
                    self.base_conhecimento.update(pickle.load(f))
            return True
        except Exception as e:
            print(f"Erro ao carregar conhecimento: {e}")
            return False
    
    def _carregar_memoria(self):
        """carrega respostas aprendidas do arquivo com tratamento de erro"""
        if os.path.exists(self.arquivo_memoria):
            try:
                with open(self.arquivo_memoria, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Erro ao carregar memória: {e}")
                return {}
        return {}
    
    def _salvar_memoria(self):
        """salva respostas aprendidas no arquivo de forma segura"""
        try:
            with open(self.arquivo_memoria, 'w', encoding='utf-8') as f:
                json.dump(self.respostas_aprendidas, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"Erro ao salvar memória: {e}")
            return False
    
    def _inicializar_vectorizador(self):
        """inicializa o vetorizador com perguntas existentes além de treinar com as respostas aprendidas"""
        if self.respostas_aprendidas:
            perguntas = list(self.respostas_aprendidas.keys())
            if perguntas:
                self.vectorizador.fit(perguntas)

    def pesquisar_duckduckgo(self, consulta):
        """faz scraping do duckduckgo para obter resultados"""
        try:
            url = f"https://html.duckduckgo.com/html/?q={consulta.replace(' ', '+')}"
            headers = {'User-Agent': 'Mozilla/5.0'}
            resposta = self.sessao.get(url, headers=headers)
            sopa = BeautifulSoup(resposta.text, 'html.parser')
            
            resultados = []
            for resultado in sopa.select('.result__body'):
                link = resultado.select_one('.result__url')['href']
                titulo = resultado.select_one('.result__title').text
                snippet = resultado.select_one('.result__snippet').text
                resultados.append({'title': titulo, 'url': link, 'snippet': snippet})
            
            return resultados[:3]  # retorna os 3 primeiros resultados
        except Exception as e:
            print(f"Erro na pesquisa DuckDuckGo: {e}")
            return []

    def pesquisar_google(self, consulta, num_resultados=5):
        """busca otimizada com tratamento de erro melhorado"""
        chave_cache = f"search_{consulta}"
        if chave_cache in self.cache:
            return self.cache[chave_cache]

        try:
            resultados = list(search(
                consulta,
                num_results=num_resultados,
                lang='pt',
                #pause=2.0
            ))
            self.cache[chave_cache] = resultados
            return resultados
        except Exception as e:
            print(f"Erro na busca por '{consulta}': {e}")
            return []

    def extrair_informacoes_chave(self, texto):
        """extrai informações principais do texto"""
        frases = sent_tokenize(texto)
        frases_importantes = []
        
        for frase in frases:
            # filtra frases muito curtas ou longas
            if 10 < len(frase.split()) < 30:
                # remoção de frases com muitos números (geralmente estatísticas)
                if len(re.findall(r'\d+', frase)) < 3:
                    frases_importantes.append(frase)
        
        return ' '.join(frases_importantes[:3])  # retorna até 3 frases principais

    def parafrasear_texto(self, texto):
        """reescreve o texto com palavras próprias do dicionario de antes"""
        palavras = texto.lower().split()
        palavras_limpas = [w for w in palavras if w not in self.palavras_parada and len(w) > 3]
        
        parafraseado = []
        for palavra in palavras:
            if palavra in self.sinonimos:
                parafraseado.append(random.choice(self.sinonimos[palavra]))
            else:
                parafraseado.append(palavra)
        
        return ' '.join(parafraseado).capitalize()

    def resumir_texto(self, texto, num_frases=2):
        """resumo do texto com fallback para nltk se sumy falhar"""
        if not texto or len(texto.split()) < 10:
            return texto[:500]
            
        try:
            parser = PlaintextParser.from_string(texto, Tokenizer("portuguese"))
            resumo = self.resumidor(parser.document, num_frases)
            return ' '.join(str(frase) for frase in resumo)
        except Exception as e:
            print(f"Erro ao resumir texto: {e}")
            frases = nltk.sent_tokenize(texto, language='portuguese')
            return ' '.join(frases[:num_frases])
    
    def _limpar_texto(self, texto):
        """limpeza avançada de texto"""
        if not texto:
            return ""
            
        try:
            texto = re.sub(r'\[\d+\]|\([^)]*\)', '', texto)
            texto = re.sub(r'\s+', ' ', texto).strip()
            
            for padrao, substituicao in self.mapa_simplificacao.items():
                texto = re.sub(padrao, substituicao, texto, flags=re.IGNORECASE)
                
            return texto
        except:
            return texto[:1000]

    def _encontrar_pergunta_similar(self, consulta):
        """encontra perguntas similares na memória com limiares ajustados"""
        if not self.respostas_aprendidas:
            return None

        perguntas = list(self.respostas_aprendidas.keys())
        if not perguntas:
            return None
        
        try:
            todas_perguntas = perguntas + [consulta]
            vetores = self.vectorizador.fit_transform(todas_perguntas)
            
            vetor_consulta = vetores[-1]
            vetores_perguntas = vetores[:-1]
            
            similaridades = cosine_similarity(vetor_consulta, vetores_perguntas)
            indice_max = np.argmax(similaridades)
            similaridade_max = similaridades[0, indice_max]

            if similaridade_max > 0.65:
                palavras_consulta = {self.stemmer.stem(w) for w in consulta.lower().split() if len(w) > 3}
                palavras_armazenadas = {self.stemmer.stem(w) for w in perguntas[indice_max].lower().split() if len(w) > 3}
                # verifica se a similaridade é alta o suficiente e se as palavras-chave coincidem
                if palavras_consulta and len(palavras_consulta & palavras_armazenadas)/len(palavras_consulta) > 0.4:
                    return perguntas[indice_max]
        except Exception as e:
            print(f"Erro ao encontrar pergunta similar: {e}")
        
        return None
    
    def aprendizado_ativo(self, pergunta):
        """mecanismo de aprendizado quando encontra perguntas similares mas n identicas"""
        similar = self._encontrar_pergunta_similar(pergunta)
        if similar and similar != pergunta.lower().strip():
            print(f"Notei que você perguntou '{pergunta}'. Devo lembrar isso como uma variação de '{similar}'?")
            entrada_usuario = input("(S/N): ").lower()
            if entrada_usuario == 's':
                self.aprender_resposta(pergunta, self.respostas_aprendidas[similar])
                return True
        return False

    def aprender_resposta(self, pergunta, resposta, sobrescrever=True):
        """armazena uma nova resposta aprendida"""
        try:
            pergunta = pergunta.lower().strip()
            if not pergunta or not resposta:
                return False
                
            if not sobrescrever and pergunta in self.respostas_aprendidas:
                return False
                
            self.respostas_aprendidas[pergunta] = resposta
            
            # atualiza o vetorizador sem perder as informações anteriores
            if hasattr(self.vectorizador, 'vocabulary_'):
                self.vectorizador.fit(list(self.respostas_aprendidas.keys()))
            else:
                self.vectorizador.fit(list(self.respostas_aprendidas.keys()))
                
            self._salvar_memoria()
            return True
        except Exception as e:
            print(f"Erro ao aprender resposta: {e}")
            return False

    def lidar_com_correcao(self, texto_correcao):
        """processa uma correção do usuario"""
        if not self.ultima_pergunta:
            return False
            
        try:
            if ':' in texto_correcao:
                correcao = texto_correcao.split(':', 1)[1].strip()
            else:
                correcao = texto_correcao.strip()
                
            if not correcao:
                return False
                
            return self.aprender_resposta(self.ultima_pergunta, correcao)
        except Exception as e:
            print(f"Erro ao processar correção: {e}")
            return False

    def _eh_dominio_confiavel(self, url):
        """verifica se o domínio é confiável com prioridade"""
        try:
            dominio = urlparse(url).netloc.lower()
            for dominio_confiavel, prioridade in self.dominios_confiaveis.items():
                if dominio_confiavel in dominio:
                    return prioridade
            return 0
        except:
            return 0

    def _conteudo_fallback(self, url):
        """retorna conteúdo padrão quando ocorre erro"""
        try:
            dominio = urlparse(url).netloc
            return {
                'content': "",
                'title': f"Não foi possível acessar o conteúdo de {dominio}",
                'url': url,
                'trust_score': 0,
                'load_time': 0
            }
        except:
            return {
                'content': "",
                'title': "Erro ao acessar conteúdo",
                'url': url,
                'trust_score': 0,
                'load_time': 0
            }

    def obter_conteudo_pagina(self, url):
        """versão com tratamento de SSL seguro e extração objetiva de conteudo"""
        if url in self.cache:
            return self.cache[url]
            
        tempo_inicio = time.time()
        resultado = self._conteudo_fallback(url)
        resultado['load_time'] = time.time() - tempo_inicio
        
        try:
            resposta = self.sessao.get(
                url,
                timeout=10,
                stream=True
            )
            resposta.raise_for_status()
            
            if 'text/html' not in resposta.headers.get('Content-Type', ''):
                return resultado
            
            sopa = BeautifulSoup(resposta.content, 'html.parser', from_encoding=resposta.encoding)
            
            for elemento in sopa(["script", "style", "nav", "footer", "iframe", 
                               "aside", "header", "form", "button", "img",
                               "comment", "noscript", "svg", "figure", "video",
                               "audio", "meta", "link", "select", "textarea"]):
                elemento.decompose()
            
            titulo = sopa.title.get_text().strip() if sopa.title else "Sem título"
            
            conteudo_principal = None
            seletores_conteudo = [
                {'name': 'article'},
                {'name': 'main'},
                {'class': 'content'},
                {'class': 'post-content'},
                {'class': 'article-body'},
                {'itemprop': 'articleBody'},
                {'id': 'content'}
            ]
            
            for seletor in seletores_conteudo:
                if conteudo_principal is None:
                    conteudo_principal = sopa.find(attrs=seletor)
            
            conteudo_principal = conteudo_principal or sopa.body
            
            elementos_texto = []
            for elemento in conteudo_principal.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'li']):
                texto = self._limpar_texto(elemento.get_text())
                if len(texto.split()) > 3:
                    elementos_texto.append(texto)
            
            conteudo = ' '.join(elementos_texto)[:5000] # limitação de caracteres
            
            resultado = {
                'content': conteudo,
                'title': titulo,
                'url': url,
                'trust_score': self._eh_dominio_confiavel(url),
                'load_time': time.time() - tempo_inicio
            }
            
            self.cache[url] = resultado
            return resultado
            
        except requests.exceptions.SSLError:
            if self._eh_dominio_confiavel(url) > 0:
                try:
                    resposta = self.sessao.get(url, timeout=10, verify=False)
                    resposta.raise_for_status()
                    print(f"Aviso: Acesso inseguro a {url} - certificado não verificado")
                except Exception as e:
                    print(f"Erro ao processar {url} (modo inseguro): {str(e)[:200]}...")
                    return resultado
            else:
                print(f"Erro de certificado SSL em {url} - domínio não confiável")
                return resultado
        except Exception as e:
            print(f"Erro ao processar {url}: {str(e)[:200]}...")
            return resultado

    def _processar_resultados_paralelo(self, consulta, resultados):
        """processa resultados em paralelo com priorização de dominios confiaveis vistos antes"""
        futures = {}
        processados = []
        
        resultados_ordenados = sorted(resultados, key=lambda url: -self._eh_dominio_confiavel(url))
        
        for url in resultados_ordenados[:5]:
            futures[self.executor.submit(self.obter_conteudo_pagina, url)] = url
        
        for future in as_completed(futures):
            url = futures[future]
            try:
                dados = future.result()
                if dados and dados['content']:
                    conteudo_minusculo = dados['content'].lower()
                    termos_consulta = consulta.lower().split()
                    relevancia = sum(termo in conteudo_minusculo for termo in termos_consulta) / len(termos_consulta)
                    dados['relevance'] = relevancia * dados['trust_score']
                    
                    processados.append(dados)
                    
                    if dados['trust_score'] >= 3 and dados['relevance'] >= 0.5:
                        return [dados]
            except Exception as e:
                print(f"Erro ao processar {url}: {e}")
        
        processados.sort(key=lambda x: (-x['relevance'], -x['trust_score']))
        return processados

    def _lidar_com_pesquisa(self, consulta_pesquisa, consulta_original):
        """método auxiliar para lidar com pesquisas genéricas"""
        resultados = self.pesquisar_google(consulta_pesquisa)
        
        if not resultados:
            resultados = self.pesquisar_google(consulta_original)
        
        if not resultados:
            resposta = {
                "response": f"Não consegui encontrar informações sobre '{consulta_original}'. Poderia reformular a pergunta?",
                "source": None
            }
            self.ultima_resposta = resposta
            return resposta
        
        dados_paginas = self._processar_resultados_paralelo(consulta_original, resultados)
        if not dados_paginas or not any(d['content'] for d in dados_paginas):
            resposta = {
                "response": f"Encontrei referências sobre '{consulta_original}', mas não consegui extrair uma explicação clara.",
                "source": None
            }
            self.ultima_resposta = resposta
            return resposta
        
        melhor_resultado = dados_paginas[0]
        conteudo = melhor_resultado['content']
        
        resumo = self.resumir_texto(conteudo)
        simplificado = self._limpar_texto(conteudo[:500])
        
        texto_resposta = f"Sobre {melhor_resultado['title']}:\n"
        texto_resposta += f"{simplificado}\n\n"
        if resumo and resumo not in simplificado:
            texto_resposta += f"Resumo: {resumo}"
        
        resposta = {
            "response": texto_resposta,
            "source": {
                "title": melhor_resultado['title'],
                "url": melhor_resultado['url']
            }
        }
        
        if len(conteudo.split()) > 30:
            self.aprender_resposta(consulta_original, simplificado, sobrescrever=False)
            
        self.ultima_resposta = resposta
        return resposta

    def aprender_e_responder(self, consulta):
        """processa a consulta e retorna uma resposta única"""
        if consulta in self.base_conhecimento:
            return random.choice(self.base_conhecimento[consulta])
        
        resultados_pesquisa = self.pesquisar_duckduckgo(consulta)
        if not resultados_pesquisa:
            return "Não consegui encontrar informações sobre isso."
        
        informacoes_combinadas = ' '.join([r['snippet'] for r in resultados_pesquisa])
        informacoes_chave = self.extrair_informacoes_chave(informacoes_combinadas)
        
        if not informacoes_chave:
            return "Encontrei resultados, mas não consegui extrair informações claras."
        
        resposta = self.parafrasear_texto(informacoes_chave)
        
        self.base_conhecimento[consulta].append(resposta)
        self.salvar_conhecimento()
        
        return resposta

    def gerar_resposta(self, consulta):
        """gera resposta com verificação de consultas curtas e fallback para erros de busca, evita 2 palavras"""
        contagem_palavras = len(consulta.split())
        if contagem_palavras <= 2:
            resposta = {
                "response": "Sua pergunta parece muito breve. Poderia elaborar ou fornecer mais detalhes? Por exemplo: em vez de 'deus grego', pergunte 'quem é o deus grego mais importante?'",
                "source": None
            }
            self.ultima_resposta = resposta
            return resposta
        
        respostas_simples = {
            'qual é o seu nome?': 'Meu nome é Pítia, e estou aqui para ajudar!',
            'quem é você?': 'Sou Pítia, sua assistente virtual. Como posso ajudar?',
            'qual é a sua função?': 'Minha função é ajudar a responder perguntas e fornecer informações.',
            'oi': 'Olá! Como posso ajudar?',
            'olá': 'Olá! Pronto para ajudar.',
            'bom dia': 'Bom dia! Como posso ser útil hoje?',
            'boa tarde': 'Boa tarde! Em que posso ajudar?',
            'boa noite': 'Boa noite! Como posso ajudar?'
        }
        
        consulta_normalizada = consulta.lower().strip()
        if consulta_normalizada in respostas_simples:
            resposta = {
                "response": respostas_simples[consulta_normalizada],
                "source": None
            }
            self.ultima_resposta = resposta
            return resposta
        
        if consulta.lower().startswith(('aprenda que ', 'lembre que ')):
            partes = consulta.split(' ', 2)
            if len(partes) == 3:
                pergunta, resposta = partes[1], partes[2]
                if self.aprender_resposta(pergunta, resposta):
                    resposta = {
                        "response": f"Entendido! Agora sei responder sobre '{pergunta}'.",
                        "source": None
                    }
                    self.ultima_resposta = resposta
                    return resposta
        
        pergunta_similar = self._encontrar_pergunta_similar(consulta)
        if pergunta_similar:
            resposta = {
                "response": self.respostas_aprendidas[pergunta_similar],
                "source": "memória"
            }
            self.ultima_resposta = resposta
            return resposta
        
        if not pergunta_similar:
            self.aprendizado_ativo(consulta)
        
        resposta_conhecimento = self.aprender_e_responder(consulta)
        if resposta_conhecimento and "não consegui" not in resposta_conhecimento.lower():
            resposta = {
                "response": resposta_conhecimento,
                "source": "base de conhecimento"
            }
            self.ultima_resposta = resposta
            return resposta
        
        filtros_site = " OR ".join(f"site:{dominio}" for dominio in self.dominios_confiaveis)
        consulta_pesquisa = f"{consulta} ({filtros_site})"
        resultados = self.pesquisar_google(consulta_pesquisa)
        
        if not resultados:
            resultados = self.pesquisar_google(consulta)
        
        if not resultados:
            resposta = {
                "response": "Não consegui encontrar informações relevantes. Poderia reformular a pergunta?",
                "source": None
            }
            self.ultima_resposta = resposta
            return resposta
        
        dados_paginas = self._processar_resultados_paralelo(consulta, resultados)
        if not dados_paginas or not any(d['content'] for d in dados_paginas):
            resposta = {
                "response": "Encontrei referências, mas não consegui extrair uma explicação clara.",
                "source": None
            }
            self.ultima_resposta = resposta
            return resposta
        
        melhor_resultado = dados_paginas[0]
        conteudo = melhor_resultado['content']
        
        resumo = self.resumir_texto(conteudo)
        simplificado = self._limpar_texto(conteudo[:500])
        
        texto_resposta = f"Sobre {melhor_resultado['title']}:\n"
        texto_resposta += f"{simplificado}\n\n"
        if resumo and resumo not in simplificado:
            texto_resposta += f"Resumo: {resumo}"
        
        resposta = {
            "response": texto_resposta,
            "source": {
                "title": melhor_resultado['title'],
                "url": melhor_resultado['url']
            }
        }
        
        if len(conteudo.split()) > 30:
            self.aprender_resposta(consulta, simplificado, sobrescrever=False)
            
        self.ultima_resposta = resposta
        return resposta

    def executar(self):
        """loop principal com aprendizado e tratamento robusto de erros"""
        print("Assistente: Olá, Sou a Pítia, sua assistente virtual. Como posso ajudar? (Digite 'sair' para encerrar)")
        while True:
            try:
                consulta = input("Você: ").strip()
                
                if not consulta:
                    continue
                    
                if consulta.lower() in {'sair', 'parar', 'adeus', 'exit', 'quit'}:
                    print("Assistente: Até logo! Estarei aqui se precisar.")
                    break
                    
                if any(consulta.lower().startswith(prefixo) for prefixo in ['correção:', 'corrija:', 'corrigir:']):
                    if self.ultima_pergunta:
                        if self.lidar_com_correcao(consulta):
                            print("Assistente: Obrigado pela correção! Vou lembrar disso.")
                        else:
                            print("Assistente: Não entendi a correção. Por favor, digite 'correção: [texto correto]'.")
                    else:
                        print("Assistente: Não há pergunta anterior para corrigir.")
                    continue
                    
                self.ultima_pergunta = consulta
                tempo_inicio = time.time()
                resultado = self.gerar_resposta(consulta)
                print(f"\nTempo de resposta: {time.time() - tempo_inicio:.2f} segundos")
                
                print(f"\nAssistente: {resultado['response']}")
                
                if resultado["source"] and resultado["source"] not in ["memória", "base de conhecimento"]:
                    print(f"\nFonte: {resultado['source']['title']}")
                    print(f"URL: {resultado['source']['url']}\n")
                
            except KeyboardInterrupt:
                print("Encerrando a assistente...")
                break
            except Exception as e:
                print(f"Erro: {e}")
                print("Ocorreu um erro inesperado. Podemos tentar novamente?")