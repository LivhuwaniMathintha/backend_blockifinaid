
using Microsoft.Extensions.Options;
using MimeKit;



namespace API.Services
{
   
    public interface IEmailService
    {
        Task SendEmailAsync(string to, string subject, string body, bool isHtml);
    }

    public class EmailService : IEmailService
    {

        private readonly MailSettings _mailSettings;
        private readonly ILogger<EmailService> _logger;
        private readonly string _email_password;
        public EmailService(IOptions<MailSettings> mailSettings, ILogger<EmailService> logger)
        {
            _mailSettings = mailSettings.Value;
            _logger = logger;
            _email_password = Environment.GetEnvironmentVariable("DOTNET_DEV_EMAIL_PASS") 
                ?? throw new InvalidOperationException("Environment variable 'DOTNET_DEV_EMAIL_PASS' is not set.");
            _mailSettings.Password = _email_password;
        }
        public async Task SendEmailAsync(string to, string subject, string body, bool isHtml)
        {
            _logger.LogInformation("Password and Username are: {Username} and {Password}", _mailSettings.Username, _mailSettings.Password);
            var email = new MimeMessage();
            email.From.Add(new MailboxAddress(_mailSettings.SenderName, _mailSettings.SenderEmail));
            email.To.Add(MailboxAddress.Parse(to));
            email.Subject = subject;

            var body_builder = new BodyBuilder();

            if (isHtml)
            {
                body_builder.HtmlBody = body;
            }
            else
            {
                body_builder.TextBody = body;
            }
            email.Body = body_builder.ToMessageBody();

            try
            {
                using (var client = new MailKit.Net.Smtp.SmtpClient())
                {
                    await client.ConnectAsync(_mailSettings.Server, int.Parse(_mailSettings.Port), MailKit.Security.SecureSocketOptions.StartTls);
                    await client.AuthenticateAsync(_mailSettings.Username, _mailSettings.Password);
                    await client.SendAsync(email);
                    await client.DisconnectAsync(true);
                }
                _logger.LogInformation("Email sent successfully to {To}", to);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to send email to {To}", to);
                throw;
            }
        }
    }

    public class MailSettings
    {
        public string? Server { get; set; }
        public string Port { get; set; }
        public string? SenderName { get; set; }
        public string? SenderEmail { get; set; }
        public string? Username { get; set; }
        public string? Password { get; set; }
    }
    
}

