import datetime


def extract_codigo_nacional_assunto(dadosBasicos):
        if 'assunto' in dadosBasicos:
            assunto = dadosBasicos['assunto']
            assunto_codigos = []

            for a in assunto:
                if isinstance(a, dict):
                    if 'codigoNacional' in a:
                        codigo = a['codigoNacional']
                    elif 'assuntoLocal' in a:
                        codigo = a['assuntoLocal']['codigoPaiNacional']
                    elif 'codigoPaiNacional' in a:
                        codigo = a['codigoPaiNacional']
                    else:
                        codigo = None
                else:
                    codigo = None

                assunto_codigos.append(codigo)
            
            if not assunto_codigos:
                return None

            return assunto_codigos
        else:
            return None


def extract_classe_processual(dadosBasicos):
    temp = dadosBasicos['classeProcessual']
    
    if temp is None:
        return None

    if isinstance(temp, int) or isinstance(temp, float):
        return int(temp)
    
    if isinstance(temp, str) and temp.isdigit():
        return int(temp)
    
    return None


def extract_orgao_julgador_nome(dadosBasicos, siglaTribunal):
    if 'orgaoJulgador' in dadosBasicos:
        if 'nomeOrgao' in dadosBasicos['orgaoJulgador']:
            return dadosBasicos['orgaoJulgador']['nomeOrgao'] + \
            ' (' + siglaTribunal + ')'
    
    return None


def extract_orgao_julgador_codigo(dadosBasicos):
    if 'orgaoJulgador' in dadosBasicos:
        if 'codigoOrgao' in dadosBasicos['orgaoJulgador']:
            return dadosBasicos['orgaoJulgador']['codigoOrgao']

    return None


def extract_orgao_julgador_instancia(dadosBasicos):
    if 'orgaoJulgador' in dadosBasicos:
        if 'instancia' in dadosBasicos['orgaoJulgador']:
            return dadosBasicos['orgaoJulgador']['instancia']

    return None


def extract_numero_processo(dadosBasicos):

    if 'numero' in dadosBasicos:
        temp = dadosBasicos['numero']

    if temp == '':
        return None

    return temp


def extract_codigo_nacional_movimento(movimento):
    if movimento is not None and isinstance(movimento, dict):
        if 'movimentoNacional' in movimento and movimento['movimentoNacional'] \
            and 'codigoNacional' in movimento['movimentoNacional']:
            codigo = movimento['movimentoNacional']['codigoNacional']
        elif 'movimentoLocal' in movimento and movimento['movimentoLocal'] \
            and 'codigoPaiNacional' in movimento['movimentoLocal']:
            codigo = movimento['movimentoLocal']['codigoPaiNacional']
        else:
            raise Exception('movimento invalido: ' + str(movimento))
        if isinstance(codigo, int):
            return codigo
        elif isinstance(codigo, str) and codigo.isdigit():
            return int(codigo)
        else:
            return None
    else:
        return None


def extract_data_hora_movimento(movimento):
    if movimento is not None and isinstance(movimento, dict):
        if isinstance(movimento['dataHora'], str):
            temp = movimento['dataHora']
        elif isinstance(movimento['dataHora'], int):
            temp = str(movimento['dataHora'])
        else:
            return None
        
        try:
            return datetime.datetime.strptime(temp, '%Y%m%d%H%M%S')
        except:
            return None 
    else:
        return None