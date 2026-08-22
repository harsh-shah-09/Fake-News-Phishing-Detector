import re
from urllib.parse import urlparse

def extract_url_features(url):
    """
    Upgraded: Extracts exactly 15 numerical features for high-accuracy ML classification.
    """
    features = []
    
    if not re.match(r'^https?://', url):
        url = 'http://' + url
        
    try:
        parsed_url = urlparse(url)
    except:
        # Fallback array must now match exactly 15 items
        return [1, 100, 1, 1, 1, 5, 1, 1, 0, 50, 5, 1, 1, 1, 20] 

    domain = parsed_url.netloc

    # --- Original 9 Features ---
    ip_pattern = re.compile(r'(([01]?\d\d?|2[0-4]\d|25[0-5])\.([01]?\d\d?|2[0-4]\d|25[0-5])\.([01]?\d\d?|2[0-4]\d|25[0-5])\.([01]?\d\d?|2[0-4]\d|25[0-5]))')
    features.append(1 if ip_pattern.search(domain) else 0)
    features.append(len(url))
    
    shorteners = re.compile(r'bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tinyurl')
    features.append(1 if shorteners.search(domain) else 0)
    
    features.append(1 if '@' in url else 0)
    features.append(1 if url.rfind('//') > 7 else 0)
    features.append(1 if '-' in domain else 0)
    features.append(domain.count('.'))
    features.append(1 if 'https' in domain else 0)
    
    trusted_domains = ['github.com', 'google.com', 'microsoft.com', 'apple.com', 'linkedin.com']
    features.append(1 if any(domain.endswith(td) for td in trusted_domains) else 0)

    # --- NEW: 6 Advanced Cybersecurity Features ---
    
    # 10. Length of the domain name specifically
    features.append(len(domain))
    
    # 11. Count of slashes (Directory depth)
    features.append(url.count('/'))
    
    # 12. Count of question marks (Query parameters)
    features.append(url.count('?'))
    
    # 13. Count of equal signs (Variable passing)
    features.append(url.count('='))
    
    # 14. Social Engineering Keywords
    suspicious_words = ['login', 'secure', 'account', 'update', 'verify', 'bank', 'confirm', 'free']
    features.append(1 if any(word in url.lower() for word in suspicious_words) else 0)
    
    # 15. Number of digits in the URL (Phishers love random number strings)
    digit_count = sum(c.isdigit() for c in url)
    features.append(digit_count)
    
    return features